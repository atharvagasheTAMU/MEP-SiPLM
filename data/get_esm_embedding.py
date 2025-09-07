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

if __name__ == "__main__":
    reference_file = pd.read_csv("../dataset/ProteinGym/reference_files/DMS_substitutions.csv")
    data_path = "../dataset/ProteinGym/substitution/"
    result_path = "../dataset/ProteinGym/representation/"
    os.makedirs(result_path,exist_ok=True)

    #set up esm model
    model,alphabet = pretrained.esm2_t33_650M_UR50D()
    model.eval() #disable dropout for deterministic results
    repr_layers = [33]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    
    for index, row in reference_file.iterrows():
        repr_dict={}
        dms_id = row["DMS_id"]
        wt_seq = row["target_seq"]
        seq_len = row["seq_len"]
        mt_seq_list = pd.read_csv(os.path.join(data_path,dms_id+".csv"))
        logits,representation = esm_encode(model,repr_layers,batch_converter,wt_seq)
        wt_seq_list = pd.read_csv(os.path.join(data_path,dms_id+".csv"))
        print(len(wt_seq_list),seq_len)
        """
        for _,wt_row in tqdm(wt_seq_list.iterrows(),total = len(wt_seq_list)):
            mutant = wt_row["mutant"]
            mt_seq = wt_row["mutated_sequence"]
            _,mt_repr = esm_encode(model,repr_layers,batch_converter,mt_seq)
            repr_dict[mutant] = mt_repr.detach().cpu().numpy()
        np.savez(os.path.join(result_path,dms_id+".npz"),**repr_dict)
        """
