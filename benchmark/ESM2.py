import torch
from esm import Alphabet, pretrained
from scipy.stats import spearmanr

import numpy as np
import pandas as pd

from tqdm import tqdm
import os

def esm_encode(model,repr_layers,tokenizer,seq):
    device = next(model.parameters()).device
    data = [("wt_protein",seq)]
    _,_,batch_tokens = tokenizer(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        result = model(batch_tokens, repr_layers, return_contacts=False)
    return result["logits"].squeeze(),result["representations"][repr_layers[0]].squeeze()[1:-1,:]

def wt_marginals(alphabet,mt_seq_list,logits):
    score_dict = {}
    for index,row in mt_seq_list.iterrows():
        mutant = row["mutant"]
        if ":" not in mutant:
            wt,idx,mt = mutant[0],int(mutant[1:-1]),mutant[-1]
        else:
            continue #only consider single site mutation for now
        wt_token,mt_token = alphabet.get_idx(wt),alphabet.get_idx(mt)
        pred_score = logits[idx-1,mt_token]-logits[idx-1,wt_token]
        score_dict[mutant] = [pred_score.detach().cpu().numpy(),row["DMS_score"]]
    return score_dict

def masked_marginals(model,repr_layers,alphabet,wt_seq,mt_seq_list):
    score_dict = {}

    device = next(model.parameters()).device
    batch_converter = alphabet.get_batch_converter()
    data = [("protein1",wt_seq)]
    _,_,batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    all_token_probs = []
    for i in tqdm(range(batch_tokens.size(1))):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0,i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = model(batch_tokens_masked)["logits"]
        all_token_probs.append(token_probs[:,i])
        masked_token_probs = torch.cat(all_token_probs,dim=0)
    for index,row in mt_seq_list.iterrows():
        mutant = row["mutant"]
        if ":" not in mutant:
            wt,idx,mt = mutant[0],int(mutant[1:-1]),mutant[-1]
        else:
            continue
        wt_token,mt_token = alphabet.get_idx(wt),alphabet.get_idx(mt)
        pred_score = masked_token_probs[idx,mt_token]-masked_token_probs[idx,wt_token]
        score_dict[mutant]=[pred_score.detach().cpu().numpy(),row["DMS_score"]]
    return score_dict
if __name__ == "__main__":
    reference_file = pd.read_csv("../dataset/ProteinGym/reference_files/DMS_substitutions.csv")
    data_path = "../dataset/ProteinGym/substitution/"
    
    #set up esm model
    model,alphabet = pretrained.esm2_t33_650M_UR50D()
    model.eval() #disable dropout for deterministic results
    repr_layers = [33]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    
    wt_spearman_list = []
    masked_spearman_list = []
    for index, row in reference_file.iterrows():
        dms_id = row["DMS_id"]
        wt_seq = row["target_seq"]
        seq_len = row["seq_len"]
        mt_seq_list = pd.read_csv(os.path.join(data_path,dms_id+".csv"))
        logits,representation = esm_encode(model,repr_layers,batch_converter,wt_seq)
        score_dict = wt_marginals(alphabet,mt_seq_list,logits)
        pred_score = [value_list[0] for value_list in score_dict.values()]
        gt_score = [value_list[1] for value_list in score_dict.values()]
        wt_marginals_spearman = spearmanr(pred_score,gt_score)[0]
        wt_spearman_list.append(wt_marginals_spearman)
        
        score_dict = masked_marginals(model,repr_layers,alphabet,wt_seq,mt_seq_list)
        pred_score = [value_list[0] for value_list in score_dict.values()]
        gt_score = [value_list[1] for value_list in score_dict.values()]
        masked_marginals_spearman = spearmanr(pred_score,gt_score)[0]
        masked_spearman_list.append(masked_marginals_spearman)
        print(dms_id,wt_marginals_spearman,masked_marginals_spearman)
