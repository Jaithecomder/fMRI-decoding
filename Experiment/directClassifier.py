import argparse
import datetime
import json
import os
import pickle
import shutil
from functools import partial
from loguru import logger

# data management
import numpy as np
import pandas as pd
from sklearn import preprocessing

# ML
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

# custom modules
from src.learning import models
from src.learning.estimators import PytorchEstimator
from src.learning.losses import multinomial_cross_entropy
from src.learning.metrics import mean_auc, recall_n
from src.utils.dataframe import dumb_tagger
from src.utils.file import get_json_files_in_dir, mkdir
from src.tools import (
    mask_rows,
    yes_or_no,
    gridsearch_complexity,
    highly_corr_cols_np,
    one_compact_line,
)
from classifierClass import trainDC, testPreds

def directClassifier(configuration="spec_template.json",
                    mode="train",
                    folder="",
                    results_file="",
                    used_gpu=0,
                    n_jobs=1,
                    verbose=False,
                    plot_training=False,
                    force=False):
    
     # ID of the experiment used in reports
    #   (filename without JSON extension)
    ID = configuration[:-5].split("/")[-1]

    logger.info(f"=== Decoding experiment {ID} ===")

    with open(configuration, encoding='utf-8') as f:
        config = json.load(f)

    # Path where to save transformed data, model and results
    path = folder + ID + "\\"
    per_label_results_file = path + "per_label_results.csv"

    # Control whether the experiment was already run
    # and results mught be overwritten
    should_ask_question_for_experiment = (
        (not force)
        &
        ((not mkdir(path)) & (mode == "train"))
    )
    if should_ask_question_for_experiment:
        if not yes_or_no("\nExperiment already run, "
                         "ARE YOU SURE you want to retrain model? "):
            print("Execution aborted")
            return 0

    # Copy original configuration to backup folder
    shutil.copy(configuration, path + configuration.split("\\")[-1])

    # If GPU is used, limit CUDA visibility to this device
    if used_gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(used_gpu)
        used_gpu = 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        used_gpu = -1

    # --------------------
    # --- DATA LOADING ---
    # --------------------
    # Metadata loading
    meta = (
        pd.read_csv(config["data"].get("meta_file"), low_memory=False, index_col=0)
        .loc[lambda df: df.kept]
    )

    # Labels vocabulary loading
    with open(config["data"]["concepts_file"], encoding='utf-8') as f:
        concept_names = [line.rstrip('\n') for line in f]
        concept_names = sorted([concept_name.strip().lower()
                         for concept_name in concept_names])

    # Features loading
    X = np.array(['../output/fmri_' + str(id) + '_resamp.nii.gz' for id in meta.index])

    # Samples' labels loading
    labels = pd.read_csv(config["data"]["labels_file"], low_memory=False, index_col=0)

    mask_labels_in_fmris = labels.index.isin(meta.index)
    labels = labels[mask_labels_in_fmris]

    mask_meta_in_labels = meta.index.isin(labels.index)
    meta = meta[mask_meta_in_labels]
    X = X[mask_meta_in_labels]

    logger.info("Data loaded")
    logger.info(f"Number of kept fMRIs in dataset {len(meta)}")

    # Only keep samples (fMRIs metadata and their embeddings) with labels
    mask_labelled = ~labels.iloc[:, 0].isna().values
    meta, X, labels = mask_rows(mask_labelled, meta, X, labels)

    # Target as a one-hot encoding of labels
    Y = dumb_tagger(labels,
                    split_regex=r",\s*",
                    vocab=concept_names,
                    label_col=None)

    # Extract vocabulary of labels present in the dataset
    vocab_orig = np.array(Y.columns)
    logger.info(f"Number of labels in the whole dataset (TRAIN+TEST): {len(vocab_orig)}")

    # Convert Y to np.array of int
    Y = Y.values * 1

    # In case the labels did not come from the proper vocabulary,
    #   remove the fmris without any label
    mask_label_checked = (Y.sum(axis=1) != 0)
    meta, X, Y = mask_rows(mask_label_checked, meta, X, Y)
    
    # Remove maps from blacklist if present
    if config["data"].get("blacklist"):
        mask_not_blacklisted = np.full(len(meta), True)
        blacklist = config["data"].get("blacklist")
        for blacklist_key in blacklist:
            mask_not_blacklisted = (
                mask_not_blacklisted
                &
                ~meta[blacklist_key].isin(blacklist[blacklist_key])
            )
        meta, X, Y = mask_rows(mask_not_blacklisted, meta, X, Y)

    logger.info(f"Number of fMRIs with labels: {len(meta)}")
    logger.info(f"Number of labels in Train: {len(vocab_orig)}")

    # Filtering labels with too few instances in train
    mask_test = (meta["collection_id"].isin(config["evaluation"]["test_IDs"]))
    colmask_lab_in_train = (Y[~mask_test].sum(axis=0)
                            >= config["labels"]["min_train"])

    number_of_rare_labels = len(vocab_orig) - int(colmask_lab_in_train.sum())
    logger.info(f"Removed {number_of_rare_labels} labels that were too rare")

    # updating X and Y
    Y = Y[:, colmask_lab_in_train]
    mask_lab_in_train = (np.sum(Y, axis=1) != 0)
    meta, X, Y = mask_rows(mask_lab_in_train, meta, X, Y)

    # updating vocab mask
    vocab_current = vocab_orig[colmask_lab_in_train]

    # Remove almost fully correlated columns
    labels_low_corr_indices = highly_corr_cols_np(Y,
                                                  vocab_current,
                                                  0.95,
                                                  True)

    number_of_too_correlated_labels = Y.shape[1] - len(labels_low_corr_indices)
    logger.info(f"Removed {number_of_too_correlated_labels} labels that were too correlated")

    Y = Y[:, labels_low_corr_indices]
    vocab_current = vocab_current[labels_low_corr_indices]

    # Update of data and testset mask after highly correlated labels removal
    mask_has_low_corr_lab = (np.sum(Y, axis=1) != 0)
    meta, X, Y = mask_rows(mask_has_low_corr_lab, meta, X, Y)
    # mask_test = meta["collection_id"].isin(config["evaluation"]["test_IDs"])
    mask_test = np.array([False] * len(meta))
    inds = np.random.choice(len(meta), int(len(meta) * 0.2), replace=False)
    mask_test[inds] = True
    
    # save original version of labels to predict before labels inference
    Y_orig = Y.copy()
    
    logger.info(f"Number of kept labels: {Y.shape[1]}")

    # Concept values transformations
    if (config["labels"].get("transformation") == "none"
            or config["labels"].get("transformation") is None):
        pass
    elif config["labels"].get("transformation") == "thresholding":
        Y = (Y >= config["labels"]["threshold"]) * 1
    elif config["labels"].get("transformation") == "normalization":
        Y = Y / Y.sum(axis=1, keepdims=True)
    else:
        raise ValueError("Unsupported transformation of concept values")

    # ------------------------------
    # --- FEATURES PREPROCESSING ---
    # ------------------------------

    # Train/valid split
    X_train, Y_train_orig, Y_train = mask_rows(~mask_test, X, Y_orig, Y)
    X_test, Y_test_orig, Y_test = mask_rows(mask_test, X, Y_orig, Y)
    indices_train = list(meta[~mask_test].index)
    indices_test = list(meta[mask_test].index)

    # ------------------------
    # --- SAMPLING WEIGHTS ---
    # ------------------------
    # Groups definition (for CV splits, loss reweighting and sampling)
    meta = (
        meta.assign(
            group=lambda df: (df["collection_id"].astype(str) + " " + df["cognitive_paradigm_cogatlas"].astype(str))
        )
    )

    # Reweighting samples if required
    weighted_samples = (
        config["torch_params"].get("group_power")
        and config["torch_params"]["group_power"] < 1.0
    )
    if weighted_samples:
        group_size = meta.groupby("group")["kept"].count()
        group_size.name = "group_size"

        # sampling weights
        group_weights = meta.join(group_size, on="group")["group_size"]
        group_weights = (group_weights ** config["torch_params"]["group_power"] / group_weights)
        sample_weights = group_weights.values.reshape((-1, 1))
        sample_weights_train = sample_weights[~mask_test]
        X_train = np.hstack((sample_weights_train, X_train))

    logger.info("Data preprocessed")
    logger.info(f"Samples kept in TRAIN: {len(X_train)}")
    logger.info(f"Samples kept in TEST: {len(X_test)}")
    if weighted_samples:
        logger.info(f"Min/Max sampling weight in TRAIN: {sample_weights_train.min()} / {sample_weights_train.max()}")

    model = trainDC(X_train, Y_train, batch_size=32, epochs=8)
    torch.save(model, path + "directCls.pt")
    # model = torch.load(path + "directCls.pt")

    # ------------------
    # --- EVALUATION ---
    # ------------------
    # Loading the Recall threshold N
    # if no N is set, we check that true labels are ranked among the first 10%
    N = config["evaluation"].get("recall@N")
    if N is None:
        N = Y_train.shape[1] // 10

    # Dumb predictions based on labels ranking in training set
    labels_rank_train = np.sum(Y_train, axis=0)
    Y_train_pred_dumb = np.tile(labels_rank_train, [len(Y_train), 1])
    Y_test_pred_dumb = np.tile(labels_rank_train, [len(Y_test), 1])

    # Remove weights from features for final prediction (1st col)
    if weighted_samples:
        X_train = X_train[:, 1:]

    # Predictions
    Y_train_pred = testPreds(model, X_train, Y_train, batch_size=32)
    Y_test_pred = testPreds(model, X_test, Y_test, batch_size=32)

    # Compute recalls for dumb and trained classifiers
    weighted_recall_n = partial(recall_n, n=N, reduce_mean=True)
    recall_train_dumb = weighted_recall_n(Y_train_pred_dumb, Y_train_orig)
    recall_test_dumb = weighted_recall_n(Y_test_pred_dumb, Y_test_orig)
    recall_train = weighted_recall_n(Y_train_pred, Y_train_orig)
    recall_test = weighted_recall_n(Y_test_pred, Y_test_orig)

    # Compute AUC for trained classifier
    auc = mean_auc(Y_test_pred, Y_test_orig)

    # Print and save performances
    perf = f"""
    PERFORMANCES:
        DUMB BASELINE:
            Recall@{str(N)} TRAIN: {str(recall_train_dumb)}
            Recall@{str(N)} TEST: {str(recall_test_dumb)}
        MODEL:
            Recall@{str(N)} TRAIN: {str(recall_train)}
            Recall@{str(N)} TEST: {str(recall_test)}
            Mean ROC AUC TEST: {str(auc)}
    """
    print(perf)

# ===================
# === MAIN SCRIPT ===
# ===================
def main():
    # --------------
    # --- CONFIG ---
    # --------------
    parser = argparse.ArgumentParser(
        description="A decoding experiment script for preprocessed fMRIs",
        epilog='''Examples:
        - python decoding_exp.py -C config_exp.json -g 0 -j 4 -v'''
    )

    parser.add_argument("-C", "--configuration",
                        default="spec_template.json",
                        help="Path of the JSON configuration file "
                             "or of a folder containing multiple configs")
    parser.add_argument("-M", "--mode",
                        default="train",  # TODO: implémenter modes
                        help="Execution mode for the experiment: "
                             "'train' or 'retrain' or 'evaluate'")
    parser.add_argument("-f", "--folder",
                        default="/home/parietal/rmenuet/work/cache/",
                        help="Path of the folder where to store "
                             "data, model and detailed results")
    parser.add_argument("-r", "--results_file",
                        default="../Data/results/decoding_perf.csv",
                        help="Path to the file where results are consolidated")
    parser.add_argument("-g", "--used_gpu",
                        type=int,
                        default=0,
                        help="GPU to be used (default 0, -1 to train on CPU)")
    parser.add_argument("-j", "--n_jobs",
                        type=int,
                        default=1,
                        help="Number of jobs "
                             "(parallel trainings during gridsearch-CV)")
    parser.add_argument("-H", "--HPC",
                        type=int,
                        default=0,
                        help="If > 0, launch the training on H nodes "
                             "on Slurm HPC using dask_jobqueue")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Activates (many) outputs")
    parser.add_argument("-y", "--force",
                        action="store_true",
                        help="Forces retraining and overwriting "
                             "of already trained model")

    args = parser.parse_args()

    configurations = get_json_files_in_dir(args.configuration)

    if args.verbose:
        print(f"Configurations used:{configurations}")

    # ----------------------
    # --- RUN EXPERIMENT ---
    # ----------------------
    for conf in configurations:
        directClassifier(configuration=conf,
                        mode=args.mode,
                        folder=args.folder,
                        results_file=args.results_file,
                        used_gpu=args.used_gpu,
                        n_jobs=args.n_jobs,
                        verbose=args.verbose,
                        force=args.force)


if __name__ == "__main__":
    # execute only if run as a script

    # fixing the seed and deterministic behavior
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # launch experiment
    main()


