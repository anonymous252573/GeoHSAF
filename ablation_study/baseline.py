# When I say baseline, I mean without matching and without interpolation
import os
import trimesh
import numpy as np
from collections import defaultdict
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.geometry.pre_shape import KendallShapeMetric
from geomstats.learning.frechet_mean import FrechetMean
import geomstats.backend as gs
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import math
import copy
from sklearn.metrics import roc_auc_score

condi_AD = 'AD'
condi_CN = 'CN'

folder_path_AD = r'C:\....\Downloads\ADNI\{}_Aligned_Offs'.format(condi_AD)
folder_path_CN = r'C:\....\Downloads\ADNI\{}_Aligned_Offs'.format(condi_CN)

# Function to compute Frobenius norm of a matrix
def frobenius_norm(matrix):
    return np.linalg.norm(matrix, ord='fro')

# Load the aligned shapes for both groups and compute their Karcher mean
def build_subject_data(folder_path):
    data = defaultdict(dict)
    for filename in os.listdir(folder_path):
        if filename.endswith('.off'):
            base = filename.replace('.off', '')
            if '_' in base:
                *subject_parts, time_point = base.split('_')
                subject_id = '_'.join(subject_parts)

                file_path = os.path.join(folder_path, filename)
                mesh = trimesh.load_mesh(file_path, process=False)
                points = np.array(mesh.vertices)

                # The Frobenius norm
                norm_points = frobenius_norm(points)
                normalized_points = points / norm_points

                data[subject_id][time_point] = normalized_points
    return data

AD_subject_data = build_subject_data(folder_path_AD)   # dict in a dict , e.g {subject_ID: {'bl': array_data, ...}, ...}
CN_subject_data = build_subject_data(folder_path_CN)

# Retain those with baseline scans
AD_subject_data_with_bl = {
    subject_id: timepoints
    for subject_id, timepoints in AD_subject_data.items()
    if 'bl' in timepoints
}
CN_subject_data_with_bl = {
    subject_id: timepoints
    for subject_id, timepoints in CN_subject_data.items()
    if 'bl' in timepoints
}

print(f"Total number of AD subjects with 'bl': {len(AD_subject_data_with_bl.keys())}")
print(f"Total number of CN subjects with 'bl': {len(CN_subject_data_with_bl.keys())}\n")

m, k = 732, 3 
shape_space = PreShapeSpace(m, k)

mean_shape_dict_AD = {}
mean_shape_dict_CN = {}
frechet_mean = FrechetMean(space=shape_space)

def compute_mean_per_timepoint(subject_data):
    timepoint_to_arrays = defaultdict(list)
    for timepoints in subject_data.values():
        for tp, array in timepoints.items():
            timepoint_to_arrays[tp].append(array)
    
    tp_to_mean = {}
    for tp, arrays in timepoint_to_arrays.items():
        stacked = gs.array(np.stack(arrays)) 
        tp_to_mean[tp] = frechet_mean.fit(stacked).estimate_
    return tp_to_mean

mean_shape_dict_AD = compute_mean_per_timepoint(AD_subject_data_with_bl)
mean_shape_dict_CN = compute_mean_per_timepoint(CN_subject_data_with_bl)

# Tangent Projection for both groups 
def tangent_subject_data(subject_data, mean_shape_dict):
    tangent_data = {}
    for subject_id, timepoints in subject_data.items():
        tangent_data[subject_id] = {}
        for timepoint, array in timepoints.items():
            mean_shape = mean_shape_dict[timepoint]
            tangent_array = shape_space.metric.log(array, mean_shape)
            tangent_data[subject_id][timepoint] = tangent_array.reshape(-1)
    return tangent_data

AD_subject_tangent_data = tangent_subject_data(AD_subject_data_with_bl, mean_shape_dict_AD)
CN_subject_tangent_data = tangent_subject_data(CN_subject_data_with_bl, mean_shape_dict_CN)

# Classification
def build_timepoint_order(subject_data):
    timepoints = set()
    for timepoints_dict in subject_data.values():
        timepoints.update(timepoints_dict.keys())

    def timepoint_to_int(tp):
        if tp == 'bl':
            return 0
        elif tp.startswith('m'):
            return int(tp[1:])
        else:
            return float('inf')  # catch-all for unknown formats

    timepoint_order = {tp: timepoint_to_int(tp) for tp in timepoints}
    return timepoint_order

def sort_timepoints(subject_data):
    timepoint_order = build_timepoint_order(subject_data)
    sorted_data = {}
    for subject_id, timepoints in subject_data.items():
        sorted_timepoints = dict(
            sorted(timepoints.items(), key=lambda item: timepoint_order[item[0]])
        )
        sorted_data[subject_id] = sorted_timepoints
    return sorted_data

AD_subject_tangent_data_sorted = sort_timepoints(AD_subject_tangent_data)
CN_subject_tangent_data_sorted = sort_timepoints(CN_subject_tangent_data)

def prepare_subject_data(group_dict, label):
    data = []
    for subject_id, time_dict in group_dict.items():
        times = sorted(time_dict.keys())
        subject_seq = []
        for t in times:
            vec = torch.tensor(time_dict[t])
            vec = vec.squeeze() 
            if vec.ndim != 1:
                raise ValueError(f"Vector at {subject_id} time {t} is not 1D after squeeze: shape={vec.shape}")
            subject_seq.append(vec)
        data.append((subject_seq, label))
    return data

CN_data = prepare_subject_data(CN_subject_tangent_data_sorted, label=1)
AD_data = prepare_subject_data(AD_subject_tangent_data_sorted, label=0)
all_data = CN_data + AD_data   # format is like this [[tensors of time_data], label], .... ]

print("Training begins !!!!")

class TimeSeriesSubjectDataset(Dataset):
    def __init__(self, data):
        self.data = data  # list of [sequence, label]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, label = self.data[idx]
        return torch.stack(sequence), label  # shape: (T_i, D), label
    
def collate_fn(batch):
    sequences, labels = zip(*batch) 

    padded = pad_sequence(sequences, batch_first=True)  # (B, T_max, D)
    lengths = [seq.size(0) for seq in sequences] 

    # Create padding mask (True = pad) - boolean mask
    mask = torch.tensor([[i >= l for i in range(padded.size(1))] for l in lengths])

    return padded.float(), torch.tensor(labels), mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes, num_heads=4, num_layers=4, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(model_dim, num_classes)

    def forward(self, x, mask):
        B, T, _ = x.size()
        x = self.input_proj(x)

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat([cls_tokens, x], dim=1) 

        # Update mask to account for cls token
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)  # (B, T+1)

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        cls_out = x[:, 0, :]  # (B, D)
        out = self.dropout(cls_out)
        out = self.fc_out(out)  # (B, num_classes)
        return out

train_data, test_data = train_test_split(all_data, test_size=0.2, stratify=[label for _, label in all_data], random_state=42)
train_dataset = TimeSeriesSubjectDataset(train_data)
test_dataset = TimeSeriesSubjectDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = train_data[0][0][0].shape[0]  
model = TransformerClassifier(input_dim=input_dim, model_dim=256, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for x, y, mask in dataloader:
        
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        optimizer.zero_grad()
        logits = model(x, mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_labels = []
    all_probs = []

    for x, y, mask in dataloader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        logits = model(x, mask)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        # **** this is for AUC *****
        probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of positive class
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')

    return avg_loss, accuracy, auc

best_test_acc = 0
best_metrics = {}

epochs = 20
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_metrics = {
            'epoch': epoch + 1,
            'acc': test_acc,
            'auc': test_auc,
            'loss': test_loss
        }

    print(f"Epoch {epoch+1}/{epochs} - "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")

print(f"\nBest Test Accuracy: {best_metrics['acc']:.4f} "
      f"(Epoch {best_metrics['epoch']}) | AUC: {best_metrics['auc']:.4f}, "
      f"Loss: {best_metrics['loss']:.4f}")