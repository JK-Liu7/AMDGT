# AMDGT: attention aware multi-modal learning using dual graph transformer for drug-disease associations prediction 

# Requirements:
- python 3.9.13
- cudatoolkit 11.3.1
- pytorch 1.10.0
- dgl 0.9.0
- networkx 2.8.4
- numpy 1.23.1
- scikit-learn 0.24.2

# Data:
The data files needed to run the model, which contain C-dataset and F-dataset.
- Drug_mol2vec: The mol2vec embeddings for drugs to construct the association network
- DrugFingerprint, DrugGIP: The similarity measurements of drugs to construct the similarity network
- DiseaseFeature: The disease embeddings to construct the association network
- DiseasePS, DiseaseGIP: The similarity measurements of diseases to construct the similarity network
- Protein_ESM: The ESM-2 embeddings for proteins to construct the association network
- DrugDiseaseAssociationNumber: The known drug disease associations
- DrugProteinAssociationNumber: The known drug protein associations
- ProteinDiseaseAssociationNumber: The known disease protein associations

# Code:
- data_preprocess.py: Methods of data processing
- metric.py: Metrics calculation
- train_DDA.py: Train the model

# Usage:
Execute ```python train_DDA.py``` 
