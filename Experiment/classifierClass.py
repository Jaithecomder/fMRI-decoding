import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from nilearn.maskers import NiftiMasker
from alive_progress import alive_bar

class DirectClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DirectClassifier, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.sigmoid(x)
        return x
    
    def predict(self, x):
        return self.forward(x).round().int()
    
def collate_fn(data, mask):
    fmriPaths = [x[0] for x in data]
    labels = np.array([x[1] for x in data])
    fmris = []
    for path in fmriPaths:
        fmri = nib.load(path)
        fmris.append(fmri)
    fmris = mask.transform(fmris)
    fmris = np.array(fmris)
    fmris = torch.tensor(fmris).float().squeeze(1)
    return fmris.cuda(), torch.tensor(labels).float().cuda()
    
def trainDC(X, Y, batch_size=8, epochs=10):
    mask = NiftiMasker(mask_img='../Data/masks/task_mask_new.nii.gz').fit()
    trainDL = DataLoader(list(zip(X, Y)), batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, mask))

    fmri = nib.load(X[0])
    input_dim = mask.transform(fmri).shape[1]
    model = DirectClassifier(input_dim, 256, Y.shape[1]).cuda()

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        losses = []
        with alive_bar(len(trainDL)) as bar:
            for i, (fmris, labels) in enumerate(trainDL):
                optimizer.zero_grad()
                outputs = model(fmris)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                bar()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses)}")
    return model

def testPreds(model, X, Y, batch_size=8):
    mask = NiftiMasker(mask_img='../Data/masks/task_mask_new.nii.gz').fit()
    testDL = DataLoader(list(zip(X, Y)), batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, mask))

    preds = []
    with torch.no_grad():
        with alive_bar(len(testDL)) as bar:
            for i, (fmris, labels) in enumerate(testDL):
                pred = model.predict(fmris)
                preds.append(pred)
                bar()
    preds = torch.cat(preds).cpu().numpy()
    return preds