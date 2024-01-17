import torch 
import torch.nn as nn
import math, pickle
from typing import Union, List
import numpy as np
from transformers import AutoTokenizer, EsmForSequenceClassification, EsmConfig
from transformers.models.esm.modeling_esm import EsmEncoder, EsmEmbeddings
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles # type: ignore
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm_model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t12_35M_UR50D", num_labels=2)
device = "cuda" if torch.cuda.is_available() else "cpu"

fp = pd.read_csv('morgan_fingerprints.csv')
fp_dict = {k:v for k,v in zip(fp['SMILES'], fp.iloc[:,2:].values)}

def get_morgan_fp(SMILES:Union[str, List[str]]):
    if isinstance(SMILES, str):
        SMILES = [SMILES]
    
    output = np.zeros((len(SMILES), 2048))
    for idx, ligand in enumerate(SMILES):
        try:
            morgan_fp = fp_dict[ligand]
        except:
            mol = MolFromSmiles(ligand)
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 3, nBits = 2048) # type: ignore
            morgan_fp = np.asarray(morgan_fp)

        output[idx,:] = morgan_fp
    
    return output

def get_tokenized_morgan_fp(SMILES:Union[str, List[str]], max_length:int=2048):
    if isinstance(SMILES, str):
        SMILES = [SMILES]
    
    output = np.zeros((len(SMILES), 2048))
    attention_mask = np.zeros((len(SMILES), 2048))
    for idx, ligand in enumerate(SMILES):
        try:
            morgan_fp = fp_dict[ligand]
        except:
            mol = MolFromSmiles(ligand)
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 3, nBits = 2048) # type: ignore
            morgan_fp = np.asarray(morgan_fp)
        
        morgan_fp = np.where(morgan_fp==1)[0]
        true_len = len(morgan_fp)
        attention_mask[idx,:true_len] = np.ones((1,true_len))

        morgan_fp = np.pad(morgan_fp, (0, max_length - true_len), 'constant', constant_values=-1) # type: ignore
        output[idx,:] = morgan_fp + 1
        

    return {
            'input_ids':torch.Tensor(output),
            'attention_mask':torch.Tensor(attention_mask)
        }

def ligand_target_collate(sample):
    ligand, protein, label = sample
    protein_inputs = tokenizer(protein, return_tensors="pt", padding='max_length', max_length=1001)
    ligand_inputs = get_tokenized_morgan_fp(ligand, max_length=2048)
    return ligand_inputs, protein_inputs, label

class ligand_targetESM(nn.Module):
    def __init__(self, config:EsmConfig, esm_model:EsmForSequenceClassification):
        super().__init__()
        self.esmEmbeddings = esm_model.esm.embeddings
        self.ligandEmbeddings = nn.Embedding(num_embeddings=2048+1, embedding_dim=256, padding_idx=0)

        self.esmEncoder = EsmEncoder(config)
        self.head = esm_model.esm.contact_head
    
    def forward(self, ligand, protein):
        ligand_emb = self.ligandEmbeddings(ligand)
        protein_emb = self.esmEmbeddings(protein)
        cat = torch.cat((ligand_emb, protein_emb))
        output = self.esmEncoder(cat)
        output = self.head(output)
        return output
