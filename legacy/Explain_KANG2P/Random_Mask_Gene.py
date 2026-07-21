#!/usr/bin/env python

import torch
import numpy as np
import pickle
import os
import argparse
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.nn.functional import relu,tanh
from torch import nn

from sklearn.metrics import confusion_matrix

# 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



import KAN

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dims, dropout):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.activation = nn.LeakyReLU()
        self.layer_norm = nn.LayerNorm(hidden_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.activation(out)
        out = self.layer_norm(out)
        out += identity
        out = self.activation(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, hidden_dims):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dims, hidden_dims)
        self.key = nn.Linear(hidden_dims, hidden_dims)
        self.value = nn.Linear(hidden_dims, hidden_dims)
        self.scale = hidden_dims ** -0.5
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = F.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
        out = torch.bmm(attn_weights, V)
        if out.size(1) == 1:
            out = out.squeeze(1)
        return out


class DualOmicsModel(nn.Module):
    def __init__(self, input_dim1, input_dim2, latent_dim, hidden_dim, dropout, num_heads=8):
        super(DualOmicsModel, self).__init__()

        # Encoder branches
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim1, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(input_dim2, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder branches
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
            SelfAttention(hidden_dim),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, input_dim1)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
            SelfAttention(hidden_dim),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, input_dim2)
        )

        # Prediction head
        self.pred = nn.Sequential(
            KAN.KANLinear(latent_dim * 2, hidden_dim).cuda(),  # Replace [KANLinear](file:///home/work/leiling/05_HumanG2P/ALS_Data/Script/KAN.py#L5-L236)
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            KAN.KANLinear(hidden_dim, 2).cuda(),
            nn.Softmax(dim=1),
        )

    def forward(self, x1, x2):
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)

        recon1 = self.decoder1(z1)
        recon2 = self.decoder2(z2)

        combined = torch.cat((z1, z2), dim=1)
        pred = self.pred(combined)

        return z1, z2, pred, recon1, recon2
    



def load_dual_omics_data(omics1_path, omics2_path):
  
    omics1_data,ttt = pickle.load(open(omics1_path, 'rb'))
    cl_name = omics1_data.columns.tolist()
    omics1_data = np.array(omics1_data)

    omics2_data = pd.read_csv(omics2_path, sep='\t', header=0)  # Adjust sep and header based on your file
    cl_name2= omics2_data.columns.tolist()
    omics2_data = omics2_data.values  # Convert to NumPy array
    omics2_data = np.array(omics2_data)

    _,dataset_Y=pickle.load(open('/home/work/leiling/05_HumanG2P/ALS_Data/A3GALT2.pkl','rb'))
    dataset_Y = np.argmax(dataset_Y, axis=1)
    test_idx = [int(line.strip()) for line in open("/home/work/leiling/05_HumanG2P/ALS_Data/test.idx", 'r')]
    y_test = dataset_Y[test_idx]
    omics1_test = omics1_data[test_idx]
    omics2_test = omics2_data[test_idx]

    omics1_df = pd.DataFrame(omics1_test, columns=cl_name)
    omics2_df = pd.DataFrame(omics2_test, columns=cl_name2)


    omics1_tensor = torch.from_numpy(omics1_test).float().to(device)
    omics2_tensor = torch.from_numpy(omics2_test).float().to(device)

    return cl_name,omics1_df,omics2_df,omics1_tensor,omics2_tensor,y_test


def predict(model, data_loader):
    """
    Use the trained model to make predictions and return latent vectors z1, z2.
    """
    model.eval()
    all_preds = []
    all_z1 = []
    all_z2 = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = [t.to(device) for t in batch]
            x1, x2 = inputs[0], inputs[1]
            z1, z2, pred, _, _ = model(x1, x2)

            all_z1.append(z1.cpu().numpy())
            all_z2.append(z2.cpu().numpy())
            _, preds = torch.max(pred, 1)
            all_preds.extend(preds.cpu().numpy())

    return (
        np.array(all_preds),
        np.concatenate(all_z1, axis=0),
        np.concatenate(all_z2, axis=0)
    )

import matplotlib.pyplot as plt


def random_select_genes(cl_name, n_genes=913):
    all_genes = set()
    for col in cl_name:
        parts = col.split(':')
        if len(parts) >= 3:
            all_genes.add(parts[1])

    all_genes = list(all_genes)
    if len(all_genes) < n_genes:
        raise ValueError(f"Only {len(all_genes)} unique genes available, less than required {n_genes}.")

    return np.random.choice(all_genes, size=n_genes, replace=False).tolist()


def apply_feature_mask(data, mask):
    masked_data = np.zeros_like(data)
    masked_data[:, mask] = data[:, mask]
    return masked_data
def main(args):
    # Step 1:
    if args.dual_omics:
        cl_name,omics1_df,omics2_df,omics1_tensor,omics2_tensor,y_test = load_dual_omics_data(
            args.omics1_path, args.omics2_path
        )
       
    else:
        data_tensor = load_data(args.data_path)[0]
        dataset = TensorDataset(data_tensor)

    # data_loader = DataLoader(dataset, batch_size=args.batch_size,shuffle=False)
    # gene_to_features = {}
    # for idx, col in enumerate(cl_name):
    #     parts = col.split(':')  # chr:gene:index
    #     if len(parts) >= 3:
    #         gene_name = parts[1]
    #         if gene_name not in gene_to_features:
    #             gene_to_features[gene_name] = []
    #         gene_to_features[gene_name].append(idx)

    # Step 2: 
    if args.dual_omics:
        model = DualOmicsModel(
            input_dim1=omics1_tensor.shape[1],
            input_dim2=omics2_tensor.shape[1],
            latent_dim=args.latent_dims,
            hidden_dim=args.hidden_dims,
            dropout=args.dropout
        ).to(device)
    else:
        model = PredictionModel(
            input_dims=data_tensor.shape[1],
            latent_dims=args.latent_dims,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout
        ).to(device)

    # Step 3: 
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"✅ Model loaded from {args.model_path}")

        # Step 5.9:  
    num_repeats = 1000
    acc_list = []

    print("🔁 Starting repeated masking experiments...")
    for i in range(num_repeats):
        print(f"🔄 Iteration {i+1}/{num_repeats}")

        # 
        selected_genes = random_select_genes(cl_name, n_genes=913)

        # 
        temp_mask = []
        for col in cl_name:
            parts = col.split(':')
            if len(parts) >= 3:
                gene_name = parts[1]
                temp_mask.append(gene_name in selected_genes)
            else:
                temp_mask.append(False)
        temp_mask = np.array(temp_mask)

        # 
        omics1_masked = apply_feature_mask(np.array(omics1_df), temp_mask)
        omics1_tensor_masked = torch.from_numpy(omics1_masked).float().to(device)
        masked_dataset = TensorDataset(omics1_tensor_masked, omics2_tensor)
        masked_loader = DataLoader(masked_dataset, batch_size=args.batch_size, shuffle=False)

        # 
        predictions_masked, _, _ = predict(model, masked_loader)

        # 
        tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_test, predictions_masked).ravel()
        acc_masked = round((tp_m + tn_m) * 1. / (tp_m + fp_m + tn_m + fn_m), 3)
        acc_list.append(acc_masked)

        print(f"ACC after masking: {acc_masked:.3f}\n")

    # 
    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)

    print(f"📊 Mean Accuracy: {mean_acc:.4f}")
    print(f"📏 Standard Deviation: {std_acc:.4f}")
    output_path = os.path.join(args.output_dir, "random_masking_accuracies.csv")
    pd.DataFrame({"Accuracy": acc_list}).to_csv(output_path, index=False)
    print(f"📊 Results saved to {output_path}")


    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a trained model.")
    parser.add_argument("--model_path", required=True, help="Path to the trained model file (.pt)")
    parser.add_argument("--dual_omics", action="store_true", help="Use dual-omics model")

    parser.add_argument("--omics1_path", help="Path to omics1 data (pkl format)")
    parser.add_argument("--omics2_path", help="Path to omics2 data (txt format)")

    parser.add_argument("--latent_dims", type=int, default=8, help="Latent dimensions of the model")
    parser.add_argument("--hidden_dims", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for prediction")
    parser.add_argument("--output_dir", help="Directory to save output files (e.g., z1.txt, z2.txt)", default="./output")


    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.dual_omics and (not args.omics1_path or not args.omics2_path):
        parser.error("For dual omics, both omics1_path and omics2_path must be provided.")

    if not args.dual_omics and not args.data_path:
        parser.error("For single omics, data_path must be provided.")

    main(args)