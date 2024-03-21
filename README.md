# BindingDB ligand target

## Viral Protein Classification
the model esm2_viral_protein.pt is fine-tuned from facebook's ESM-2 to predict whether a protein comes from virus or not from their amino acid sequence. the model is trained on proteins found in <a href='https://library.ucsd.edu/dc/object/bb6496315b'>BindingDB ligand target dataset</a>

## Ligand-Target Binding Affinity Prediction
A classification model predicting if the IC50 value of a small molecule ligand and protein pair is below certain threshold (10 nM).

![image](https://github.com/warastra/ligand_target_prediction/assets/36398445/a3580417-a402-4c0c-9f07-4775a4777f12)


# References
ESM-2
```
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Sal and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```
