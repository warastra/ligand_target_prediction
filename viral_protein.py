import torch 
import torch.nn as nn
import math
from typing import Union, List
import numpy as np
from transformers import AutoTokenizer, EsmForSequenceClassification


device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

def esm_collate(sample):
    inputs, labels = sample
    inputs = tokenizer(inputs, return_tensors="pt", padding='max_length', max_length=1000)
    return inputs, labels

def train(model, dataloader):
    model.train()
    model.to(device)
    epochs = 100
    total_loss = 0
    optim = torch.optim.AdamW(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        total_loss = 0
        for batch, labels in dataloader:
            optim.zero_grad()
            # input = batch.clone()
            inputs = tokenizer(batch, return_tensors="pt", padding='max_length', max_length=1001)
            # print(inputs, labels)

            loss = model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device), labels=labels.to(device)).loss # type: ignore
            total_loss += loss
            loss.backward()
            optim.step()
    
        # if (epoch+1)%40==0 or epoch==0:
        print("Epoch: {} -> loss: {}".format(epoch+1, total_loss/(len(dataloader))))

def test(model, dataloader):
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        test_correct = 0
        total_loss = 0
        for batch, labels in dataloader:
            # input = batch.clone()
            inputs = tokenizer(batch, return_tensors="pt", padding='max_length', max_length=1001)
            # print(inputs, labels)
            output = model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device), labels=labels.to(device)) # type: ignore
            pred = torch.argmax(output.logits, dim=1)
            test_correct += (pred.cpu() == labels).float().sum()

            loss = output.loss
            total_loss += loss.cpu()
        accuracy = test_correct / len(dataloader)
    return total_loss, accuracy

def load_pretrained_esm(
        saved_model_name:str='esm2_viral_protein.pt', 
        base_esm_model_name:str='facebook/esm2_t12_35M_UR50D',
        n_label:int=2
        ):
    checkpoint = torch.load(saved_model_name)
    # esm_model = EsmForSequenceClassification(EsmConfig())
    esm_model = EsmForSequenceClassification.from_pretrained(base_esm_model_name, num_labels=n_label)

    esm_model.load_state_dict(checkpoint['model_state_dict']) # type: ignore
    return esm_model

def make_pred(chain_seq:List[str], model):
    inputs = tokenizer(chain_seq, return_tensors="pt", padding='max_length', max_length=1001)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1)
    return pred, logits