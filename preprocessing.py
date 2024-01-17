from typing import List, Union 
import numpy as np, pickle
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder

with open('amino_acid_tokens.pkl', 'rb') as f:
    amino_acid_tokens = pickle.load(f)

class proteinDataset(Dataset):
    def __init__(self, src, label=None, label_type='classification'):
        self.src = src

        if label is not None and label_type=='classification':
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.label = encoder.fit_transform(label.reshape(-1, 1))
        
        elif label is not None and label_type=='regression':
            self.label = label

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        if self.label is not None:
            label = self.label[idx]
            return src, label
        else:
            return src

class protein_ligand_Dataset(Dataset):
    def __init__(self, ligandSMILES, proteinSeq, label, label_type='classification'):
        self.ligandSMILES = ligandSMILES
        self.proteinSeq = proteinSeq

        if label is not None and label_type=='classification':
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.label = encoder.fit_transform(label.reshape(-1, 1))
        
        elif label is not None and label_type=='regression':
            self.label = label

    def __len__(self):
        return len(self.ligandSMILES)

    def __getitem__(self, idx):
        ligand = self.ligandSMILES[idx]
        protein = self.proteinSeq[idx]
        label = self.label[idx]
        return ligand, protein, label



def custom_chain_tokenizer(texts, max_length:int, return_tensors:str):
    tokenized = np.zeros((len(texts), max_length))
    for idx, text in enumerate(texts):
        chain_id = [amino_acid_tokens[x] for x in text.strip().lower()]
        for id_idx, id in enumerate(chain_id):
            tokenized[idx][id_idx] = id

    return torch.tensor(tokenized)




