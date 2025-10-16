import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics

from data.argoverse.argo_csv_dataset import ArgoCSVDataset
from data.argoverse.utils.torch_utils import collate_fn_dict
from model.tcrat_pred import TcratPred


# Make newly created directories readable, writable and descendible for everyone (chmod 777)
os.umask(0)

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser()
parser = TcratPred.init_args(parser)

parser.add_argument("--split", choices=["val", "test"], default="val")
parser.add_argument("--ckpt_path", type=str,
                    default="/path/to/checkpoint.ckpt")


def main():

    args = parser.parse_args()

    if args.split == "val":
        dataset = ArgoCSVDataset(args.val_split, args.val_split_pre, args)
    else:
        dataset = ArgoCSVDataset(args.test_split, args.test_split_pre, args)

    data_loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        num_workers=args.val_workers,
        collate_fn=collate_fn_dict,
        shuffle=False,
        pin_memory=True,
    )

    # Load model with pretrained weights
    model = TcratPred.load_from_checkpoint(checkpoint_path=args.ckpt_path)
    model.eval()

    # Containers for output
    predictions = dict()
    gts = dict()
    cities = dict()
    probabilities = dict()
    
    # Inference over dataset
    for data in tqdm(data_loader):
        data = dict(data)
        
        # Disable gradient computation for inference
        with torch.no_grad():
            output = model(data)
            # Extract only first prediction per batch element and move to CPU
            output = [x[0:1].detach().cpu().numpy() for x in output]
            
        # Process each element in batch
        for i, (argo_id, prediction) in enumerate(zip(data["argo_id"], output)):
            pred = prediction.squeeze()
            predictions[argo_id] = pred
            
            # Compute probability normalization manually (post-softmax adjustment)
            sum_1 = np.sum(prediction.squeeze(), axis=1)
            sum_2 = np.sum(sum_1, axis=1)
            sotmax_out = softmax(sum_2)
            
            # Ensure sum of probabilities = 1 (manual correction for drift)
            sum_soft = np.sum(sotmax_out)
            if sum_soft > 1:
                index_max = np.argmax(sotmax_out, axis=0)
                sotmax_out[index_max] -= (sum_soft - 1)
            elif sum_soft < 1:
                index_min = np.argmin(sotmax_out, axis=0)
                sotmax_out[index_min] += (1 - sum_soft)

            probabilities[argo_id] = sotmax_out
            cities[argo_id] = data["city"][i]
            gts[argo_id] = data["gt"][i][0]  # Only store GT if in validation mode

    # Evaluate or export results
    results_6 = compute_forecasting_metrics(
        predictions, gts, cities, 6, 60, 2, probabilities)
    results_1 = compute_forecasting_metrics(
        predictions, gts, cities, 1, 60, 2, probabilities)
        

if __name__ == "__main__":
    main()
