# Protein Mutation Effect Prediction using structure information and protein language model

## Prerequisites:
Setting up environment
```
conda create -n mep_env python=3.10
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/facebookresearch/esm.git #newest version needed for ESM-IF (https://github.com/facebookresearch/esm/pull/386)
pip install transformers
pip install scipy
pip install pandas
pip install h5py

#optional: only needed for GearNet
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install torchdrug
pip install easydict pyyaml

#optional: only needed for SaProt
conda install -c conda-forge -c bioconda foldseek

#optional: only needed for ESM-IF
pip install torch_geometric
pip install biotite==0.41.1
```
## Installation
```
git clone --branch update https://github.com/yxliu-TAMU/MEP-SiPLM

Download dataset from Zenodo (https://zenodo.org/records/10976493)
```

## File tree
```
--benchmark: scripts to evaluate the previous models performance\
--data: scripts to preprocess the dataset\
--dataset: ProteinGym dataset and related files. (unzip from Zenodo link)
```

## Zero-shot Mutation Effect Inference
```
conda activate mep_env
cd benchmark
python ESM2.py # ESM2 zero-shot mutation effect prediction using wild-type marginal and masked marginal methods: https://huggingface.co/blog/AmelieSchreiber/mutation-scoring
```
## Precomputer Embedding

Not needed if you download the dataset from Zenodo in the Installation Step
File list can be found: https://drive.google.com/drive/folders/1xB43lm6M-MuwqP4KLqEruJ4GBIuLURQY?usp=sharing
```
cd data
python get_esm_embedding.py --file_list job_12h_1_files.txt [choose from 1 to 4, each will spend around 15h using A100]
python get_SaProt_embedding.py --file_list job_12h_1_files.txt [choose from 1 to 4]
python get_GearNet_embedding.py
python get_esm_IF_embedding.py
```
## Model Training
```
cd supervised
python train.py --embedding_list esm2 esm_if gearnet --test_fold 0 --ckpt_path ../ckpt/esm2_struc/fold0/ #embedding list choose from: [saprot, esm2, esm_if, gearnet], test_fold: [0,1,2,3,4]
```
## Evaluation
```
python evaluation.py  --embedding_list esm2 esm_if gearnet --test_fold 0 --ckpt_path ../ckpt/esm2_struc/fold0/mlp_best_fold0.pt --dms_csv ../dataset/ProteinGym/substitution_split/A0A1I9GEU1_NEIME_Kennouche_2019.csv #make sure embedding list is the same as your training list
```
## To Do:
```
1. 7 proteins' sequence and structure not match: seq_id: {A0A140D2T1_ZIKV_Sourisseau_2019, BRCA2_HUMAN_Erwood_2022_HEK293T, CAS9_STRP1_Spencer_2017_positive, P53_HUMAN_Giacomelli_2018_Null_Etoposide, P53_HUMAN_Giacomelli_2018_Null_Nutlin, P53_HUMAN_Giacomelli_2018_WT_Nutlin,
POLG_HCVJF_Qi_2014,}. skipped them for now.

2. Several sequence have multi-mutation sequences. Skipped them for now.
```
