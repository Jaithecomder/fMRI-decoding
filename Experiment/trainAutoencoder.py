import pandas as pd
import pickle as pkl
import nibabel as nib
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from alive_progress import alive_bar
from nilearn.maskers import NiftiMasker

from autoencoder import Autoencoder

torch.cuda.set_per_process_memory_fraction(0.8)

meta = pd.read_csv('../Data/meta/fmris_meta.csv', low_memory=False, index_col=0).loc[lambda df: df.kept]
ids = meta.index

fmriPaths = ['../output/fmri_' + str(id) + '_resamp.nii.gz' for id in ids]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

mask = NiftiMasker(mask_img='../Data/masks/task_mask_new.nii.gz').fit()

def collate_fn(fmriPaths):
    fmris = []
    for path in fmriPaths:
        fmri = nib.load(path).get_fdata()
        fmris.append(fmri)
    fmris = np.array(fmris)
    fmris = torch.tensor(fmris).float().unsqueeze(1)
    return fmris

trainDL = DataLoader(fmriPaths, batch_size=4, shuffle=True, collate_fn=collate_fn)
input_dim = nib.load(fmriPaths[0]).get_fdata().shape

autoencoder = Autoencoder(input_dim, 256, dropout=0.15).to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

for epoch in range(8):
    losses = []
    with alive_bar(len(trainDL)) as bar:
        for i, fmris in enumerate(trainDL):
            optimizer.zero_grad()
            fmris = fmris.to(device)
            outputs = autoencoder(fmris)
            loss = torch.nn.functional.mse_loss(outputs, fmris)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            bar()
    print('Epoch:', epoch, 'Loss:', np.mean(losses))

torch.save(autoencoder.state_dict(), './store/autoencoder.pt')
# autoencoder.load_state_dict(torch.load('./store/autoencoder.pt'))

dataDL = DataLoader(fmriPaths, batch_size=8, shuffle=False, collate_fn=collate_fn)

print('Generating embeddings...')
embeddings = torch.zeros((len(fmriPaths), 256))
with torch.no_grad():
    with alive_bar(len(dataDL)) as bar:
        for i, fmris in enumerate(dataDL):
            fmris = fmris.to(device)
            outputs, _ = autoencoder.encode(fmris)
            embeddings[i * 8:i * 8 + len(outputs)] = outputs
            bar()

embeddings = embeddings.numpy()
with open('./fmris_emb_ae.p', 'wb') as f:
    pkl.dump(embeddings, f)    