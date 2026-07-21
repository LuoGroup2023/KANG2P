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
import sys
from scipy.stats import pearsonr

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
            KAN.KANLinear(hidden_dim, 1).cuda(),
            
        )

    def forward(self, x1, x2):
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)

        recon1 = self.decoder1(z1)
        recon2 = self.decoder2(z2)

        combined = torch.cat((z1, z2), dim=1)
        pred = self.pred(combined)
        pred = pred.squeeze()

        return z1, z2, pred, recon1, recon2
    



def load_data(args):
    if args.load_from_pkl:
        # 直接从 pkl 加载数据
        print(f"🔄 Loading data from existing pickle file: {args.load_from_pkl}")
        with open(args.load_from_pkl, 'rb') as f:
            omics1_data, omics2_data, y_data = pickle.load(f)
    else:
        # 否则从原始 txt 文件中读取并保存为 pkl
        print("📄 Reading from raw text files...")
        df_omics1 = pd.read_csv(args.feature_file, sep='\t', header=0)
        omics1_data = df_omics1.iloc[:, 1:].values

        df_omics2 = pd.read_csv(args.tr_file, sep='\t', header=0)
        omics2_data = df_omics2.iloc[:, 1:].values

        df_y = pd.read_csv(args.y_file, sep='\t', header=0)
        y_data = df_y.iloc[:, 1].values

        omics1_data = np.array(omics1_data)
        omics2_data = np.array(omics2_data)
        y_data = np.array(y_data)

        output_pkl_path = os.path.join(args.output_dir, 'dataset.pkl')
        with open(output_pkl_path, 'wb') as f:
            pickle.dump((omics1_data, omics2_data, y_data), f)
        print(f"✅ Saved datasets to {output_pkl_path}")

    return omics1_data, omics2_data, y_data


def predict(model, data_loader):
    """
    使用训练好的模型进行回归预测
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = [t.to(device) for t in batch]
            x1, x2 = inputs[0], inputs[1]
            _, _, pred, _, _ = model(x1, x2)
            all_preds.extend(pred.cpu().numpy())

    return np.array(all_preds)


def main(args):
    # Step 1: 加载数据
    if args.dual_omics:
        omics1_data, omics2_data, dataset_Y = load_data(args)
        test_idx = [int(line.strip()) for line in open(args.test_idx_file, 'r')]

        omics1_test = omics1_data[test_idx]
        omics2_test = omics2_data[test_idx]
        y_test = dataset_Y[test_idx]
        omics1_test_tensor = torch.from_numpy(omics1_test).float().to(device)
        omics2_test_tensor = torch.from_numpy(omics2_test).float().to(device)
        y_test_tensor = torch.from_numpy(y_test).float().to(device)
        test_dataset = TensorDataset(
        omics1_test_tensor, omics2_test_tensor, y_test_tensor)

    
        testloader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
    else:
        data_tensor = load_data(args.data_path)[0]
        dataset = TensorDataset(data_tensor)

    

    # Step 2: 加载模型
    if args.dual_omics:
        model = DualOmicsModel(
            input_dim1=omics1_test_tensor.shape[1],
            input_dim2=omics2_test_tensor.shape[1],
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

    # Step 3: 加载预训练权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"✅ Model loaded from {args.model_path}")

    # Step 4: 进行预测
    predictions = predict(model, testloader)
    # Step 5: 计算 PCC
    pcc, _ = pearsonr(predictions, y_test)
    print(f"Pearson Correlation Coefficient (PCC): {pcc:.4f}")

    # Step 6: 创建 DataFrame 并保存为 CSV
    df_results = pd.DataFrame({
        'Predicted': predictions,
        'True_label': y_test
    })

    output_csv = args.output_file if args.output_file else "predictions.csv"
    df_results.to_csv(output_csv, index=False,sep='\t')
    print(f"📊 Predictions saved to {output_csv}")
    

    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a trained model.")
    parser.add_argument("--model_path", required=True, help="Path to the trained model file (.pt)")
    parser.add_argument('--load_from_pkl', type=str, default=None,
                    help='If provided, load data directly from this .pkl file instead of reading raw files')
    parser.add_argument("--dual_omics", action="store_true", help="Use dual-omics model")
    parser.add_argument('--test_idx_file', type=str, required=False,
                        default="",
                        help='Path to test index file')

    parser.add_argument("--latent_dims", type=int, default=8, help="Latent dimensions of the model")
    parser.add_argument("--hidden_dims", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for prediction")
    parser.add_argument("--output_file", help="File path to save predictions (optional)")

    args = parser.parse_args()


    main(args)