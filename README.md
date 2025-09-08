# Protein Mutation Effect Prediction using structure information and protein language model

## Prerequisites:
Setting up environment
```
conda create -n mep_env python=3.10 pytorch=2.3.1
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/facebookresearch/esm.git #newest version needed for ESM-IF (https://github.com/facebookresearch/esm/pull/386)
pip install transformers
pip install scipy
pip install pandas

#optional: only needed for GearNet
pip install torch-scatter torch-cluster -f https://pytorch-geometric.com/whl/torch-2.3.1+cu122.html
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

Download dataset from Zenodo ([zenodo](https://zenodo.org/records/10951915?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjdmNDkzYjdjLWY3YzUtNGE1MC1hMGZhLWYyYmRkZWVkMDllMyIsImRhdGEiOnt9LCJyYW5kb20iOiJjMmM2MzVmZTY1YWYyY2JlYTE1YjBkMGI0NWJjNmQ3YSJ9.hx6zOm4OM-RnW4iMSUUlGulEhFbm5uCG3wT48V60nngr-a5dwEd7Z6sITZM7R2age66kDCQON3L3pXLZWccXgg))
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
## To Do:
```
1. 7 proteins' sequence and structure not match: seq_id: {A0A140D2T1_ZIKV_Sourisseau_2019, BRCA2_HUMAN_Erwood_2022_HEK293T, CAS9_STRP1_Spencer_2017_positive, P53_HUMAN_Giacomelli_2018_Null_Etoposide, P53_HUMAN_Giacomelli_2018_Null_Nutlin, P53_HUMAN_Giacomelli_2018_WT_Nutlin,
POLG_HCVJF_Qi_2014,}. skipped them for now.

2. Several sequence have multi-mutation sequences. Skipped them for now.
```
