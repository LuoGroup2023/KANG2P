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

# 设置设备
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')



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
            KAN.KANLinear(latent_dim * 2, hidden_dim).to(device),  # Replace [KANLinear](file:///home/work/leiling/05_HumanG2P/ALS_Data/Script/KAN.py#L5-L236)
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            KAN.KANLinear(hidden_dim, 2).to(device),
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
    """
    load omics data
    """
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

    return omics1_df,omics2_df,omics1_tensor,omics2_tensor,y_test


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


def compute_latent_gradients(model, z1, z2):
    """
    
    """
    z1.requires_grad = True
    z2.requires_grad = True

    combined = torch.cat((z1, z2), dim=1)
    pred = model.pred(combined)

    # 假设我们关注类别 1 的概率
    pred[:, 1].sum().backward()

    return z1.grad.cpu().numpy(), z2.grad.cpu().numpy()


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))  # 
    return e_x / e_x.sum(axis=0)
def main(args):
    # Step 1: load data
    if args.dual_omics:
        omics1_df,omics2_df,omics1_tensor,omics2_tensor,y_test = load_dual_omics_data(
            args.omics1_path, args.omics2_path
        )
        dataset = TensorDataset(omics1_tensor, omics2_tensor)
    else:
        data_tensor = load_data(args.data_path)[0]
        dataset = TensorDataset(data_tensor)

    data_loader = DataLoader(dataset, batch_size=args.batch_size,shuffle=False)

    # Step 2: load model
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

    # Step 3: load pre-trained weight
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"✅ Model loaded from {args.model_path}")

    # Step 4: predict
    predictions, z1_array, z2_array = predict(model, data_loader)
    # Step 5: analyse Z1 和 Z2 's contribution
    z1_tensor = torch.tensor(z1_array).float().to(device)
    z2_tensor = torch.tensor(z2_array).float().to(device)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    acc = round((tp + tn) * 1. / (tp + fp + tn + fn),3)
    precision = round(tp*1./(tp+fp),3)
    recall = round(tp*1./(tp+fn),3)
    f1=round(2*(precision*recall)/(precision+recall),3)

    print('\t'.join(list(map(str,[precision,recall,f1,acc])))+'\n')
    # Step 6: Analyzing Z1 contributions via multi-layer KAN activation functions
    print("📊 Analyzing Z1 contributions via multi-layer KAN activation functions...")

    layer1 = model.pred[0]
    layer2 = model.pred[4]

    x_combined = torch.cat([z1_tensor, z2_tensor], dim=1)

    with torch.no_grad():
        # Layer 1
        bases1 = layer1.b_splines(x_combined)
        weight1 = layer1.scaled_spline_weight.permute(0, 2, 1)
        contribution1 = torch.einsum('bik,oki->boi', bases1, weight1).mean(dim=0)  # (H, I1)

        # Layer 2
        x_hidden = layer1(x_combined)
        bases2 = layer2.b_splines(x_hidden)
        weight2 = layer2.scaled_spline_weight.permute(0, 2, 1)
        contribution2 = torch.einsum('bik,oki->boi', bases2, weight2).mean(dim=0)  # (2, H)

        # total contribution
        total_contribution = contribution2 @ contribution1  # (2, I1+I2)
        latent_dim = z1_tensor.shape[1]
        z1_contributions = total_contribution[:, :latent_dim].abs().mean(dim=0)  # (I1, )

    
    np.savetxt(os.path.join(args.output_dir, "z1_contribution_kan.txt"), z1_contributions.cpu().numpy(), delimiter="\t")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(z1_contributions)),z1_contributions.cpu().numpy(),color='skyblue')
    plt.title("Z1 Dimension Contribution to Phenotype via Multi-layer KAN Activation")
    plt.xlabel("Z1 Dimension Index")
    plt.ylabel("Contribution Score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "z1_contribution_kan.png"))
    plt.close()
    


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