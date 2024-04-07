# fMRI Decoding

Code for the article "Comprehensive decoding mental processes from Web repositories of functional brain images", published in [Nature Scientific Reports 2022](https://www.nature.com/articles/s41598-022-10710-1).

## Installation

- Setup a virtual environment (conda, virtualenv, pyenv, ..) and install requirements
    ```bash
    virtualenv venv -p python3.6
    . venv/bin/activate
    pip install -r requirements.txt
    ```
- If you have Neurovault already downloaded on your machine, create a symlink in the `Data/` folder
    ```bash
    sudo ln -s /path/to/already/downloaded/neurovault Data/neurovault
    ```
  _Note:_ For Parietal users, Neurovault is already downloaded at `/storage/store2/data/neurovault` on Drago.
- Add folder path to the PYTHONPATH
    ```bash
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    ```

## Experiments

All scripts to perform the experiments are stored in the `Experiments/` folder. In particular:

- the `a_preparation_pipeline` script performs all preprocessing for images and labels
- the `b_decoding_experiment` script performs training (from the output of the previous scripts)
- the `c_plot_perf_and_models` generate some figures of the paper

For explanations on each argument, you can call `python {a_preparation_pipeline.py|b_decoding_experiment.py|c_plot_perf_and_models.py} --help`
