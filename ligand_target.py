import torch 
import torch.nn as nn
import math, pickle
from typing import Union, List, Tuple
import numpy as np
from transformers import AutoTokenizer, EsmForSequenceClassification, EsmConfig
from transformers.models.esm.modeling_esm import EsmEncoder, EsmEmbeddings
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles # type: ignore
from utils.constants import SMILES_dict

import pandas as pd
import warnings

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm_model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t12_35M_UR50D", num_labels=2)
device = "cuda" if torch.cuda.is_available() else "cpu"

fp = pd.read_csv('morgan_fingerprints.csv')
fp_dict = {k:v for k,v in zip(fp['SMILES'], fp.iloc[:,2:].values)}

def get_morgan_fp(SMILES:Union[str, List[str]], as_tensor=False):
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
    if as_tensor:
        return torch.Tensor(output).type(torch.IntTensor) # type: ignore
    else:
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
            'input_ids':torch.Tensor(output).type(torch.IntTensor), # type: ignore
            'attention_mask':torch.Tensor(attention_mask).type(torch.IntTensor) # type: ignore
        }

def ligand_protein_tokenizer(protein_Seq, ligand_SMILES, max_length:int=1422, return_tensors:str='pt'):
    tokenizedP = tokenizer(protein_Seq, return_tensors=return_tensors, padding='max_length', max_length=1001)
    ligand_maxlen = max_length - 1001

    tokenizedL = np.ones((len(ligand_SMILES), ligand_maxlen))
    att_maskL = np.zeros((len(ligand_SMILES), ligand_maxlen))
    for idx, text in enumerate(ligand_SMILES):
        att_maskL[idx][:len(text)] = np.ones((len(text)))
        smiles_id = [SMILES_dict[x] for x in text]
        for id_idx, id in enumerate(smiles_id):
            tokenizedL[idx][id_idx] = id
    tokenizedL = torch.tensor(tokenizedL)
    att_maskL = torch.tensor(att_maskL)

    print(tokenizedL.shape, tokenizedP['input_ids'].shape)
    tokenized = torch.cat((tokenizedP['input_ids'], tokenizedL), dim=1).type(torch.IntTensor)
    att_mask = torch.cat((tokenizedP['attention_mask'], att_maskL), dim=1).type(torch.IntTensor)

    return {
        'input_ids':tokenized,
        'attention_mask':att_mask
    }

def ligand_target_collate(sample):
    ligand, protein, label = sample
    protein_inputs = tokenizer(protein, return_tensors="pt", padding='max_length', max_length=1001)
    ligand_inputs = get_tokenized_morgan_fp(ligand, max_length=2048)
    return ligand_inputs, protein_inputs, label

def get_extended_attention_mask(
       attention_mask: torch.Tensor, device: torch.device = None, dtype: torch.float = None # type: ignore
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

class ligand_targetESM(nn.Module):
    def __init__(self, config:EsmConfig, esm_model:EsmForSequenceClassification):
        super().__init__()
        self.esmEmbeddings = esm_model.esm.embeddings
        self.ligandEmbeddings = nn.Embedding(num_embeddings=2048+1, embedding_dim=256, padding_idx=0)

        self.esmEncoder = EsmEncoder(config)
        self.head = esm_model.esm.contact_head
    
    # def forward(self, ligand, protein):
    #     ligand_emb = self.ligandEmbeddings(ligand)
    #     protein_emb = self.esmEmbeddings(protein)
    #     protein_emb = self.esmEncoder(protein_emb)
    #     cat = torch.cat((ligand_emb, protein_emb))
    #     output = self.head(cat)
    #     return output
    
    def forward(self, ligand, protein):
        ligand_emb = self.ligandEmbeddings(ligand)
        protein_emb = self.esmEmbeddings(protein)
        cat = torch.cat((ligand_emb, protein_emb))
        output = self.esmEncoder(cat)
        output = self.head(cat)
        return output
