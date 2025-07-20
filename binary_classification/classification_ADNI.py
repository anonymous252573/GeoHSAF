import torch
import gpytorch
import os
import pickle
import numpy as np
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

data_folder_AD = r'C:\....\ADNI\After_matching\{}_PCA_tangent_data'.format(condi_AD) 
data_folder_CN = r'C:\....\ADNI\After_matching\{}_PCA_tangent_data'.format(condi_CN) 

data_folder_interp_AD = r'C:\....\ADNI\After_matching\{}_interpolated_PCA_tangent_data'.format(condi_AD) 
data_folder_interp_CN = r'C:\....\ADNI\After_matching\{}_interpolated_PCA_tangent_data'.format(condi_CN) 

def extract_time_map(folder):
    time_labels = set()
    for file in os.listdir(folder):
        if not file.endswith('.pkl'):
            continue
        with open(os.path.join(folder, file), 'rb') as f:
            data = pickle.load(f)
            time_labels.update(data.keys())

    # Convert to numeric time values
    def convert(label):
        if label == 'bl':
            return 1
        elif label.startswith('m') and label[1:].isdigit():
            return int(label[1:])
        else:
            return None  # Skip invalid labels
    
    valid_labels = [label for label in time_labels if convert(label) is not None]
    time_map = {label: convert(label) for label in sorted(valid_labels, key=convert)}
    return time_map    #bl = 1, m03 = 3, m06= 6, m12 = 12 , etc; returns {'bl': 1, 'm03': 3, ....}  

# **** Let's collect all available timepoints in a dictionary for both groups ****
time_map_AD = extract_time_map(data_folder_AD)
time_map_CN = extract_time_map(data_folder_CN)

# ***** We load the interpolated data for each subject ****
def load_interpolated_subject_time_series(folder_path, time_map):
    subject_data = {}

    for filename in os.listdir(folder_path):
        if not filename.endswith(".pkl"):
            continue

        parts = filename.split('_')
        if 'bl' not in parts:
            continue

        bl_index = parts.index('bl')
        subject_id = '_'.join(parts[:bl_index])

        timepoints_str = filename.replace('.pkl', '').split('_')[bl_index:]
        if 'interpolated' in timepoints_str:
            timepoints_str.remove('interpolated')

        valid_times = [tp for tp in timepoints_str if tp in time_map]
        if not valid_times:
            continue

        min_time = time_map[valid_times[0]]
        max_time = time_map[valid_times[-1]]

        selected_keys = [k for k, v in time_map.items() if min_time <= v <= max_time]

        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        filtered_data = {k: data[k] for k in selected_keys if k in data}
        subject_data[subject_id] = filtered_data

    return subject_data

subject_AD_dict = load_interpolated_subject_time_series(data_folder_interp_AD, time_map_AD)  #values of this dict are also dict {ID: {time: data, ..., .....}}
subject_CN_dict = load_interpolated_subject_time_series(data_folder_interp_CN, time_map_CN)  

print("Done Loading Data for Training !!!!")

def get_first_array_dimension(d):
    for subj in d:
        for t in d[subj]:
            return d[subj][t].shape[-1]
    raise ValueError("Empty dictionary")

def pad_dict_arrays(d, target_dim):
    for subj in d:
        for t in d[subj]:
            arr = d[subj][t]
            if arr.shape[-1] < target_dim:
                pad_width = target_dim - arr.shape[-1]
                d[subj][t] = np.pad(arr, ((0, 0), (0, pad_width)) if arr.ndim == 2 else ((0, pad_width),), mode='constant')
    return d

dim1 = get_first_array_dimension(subject_AD_dict)
dim2 = get_first_array_dimension(subject_CN_dict)

max_dim = max(dim1, dim2)
if dim1 < max_dim:
    subject_AD_dict = pad_dict_arrays(subject_AD_dict, max_dim)
if dim2 < max_dim:
    subject_CN_dict = pad_dict_arrays(subject_CN_dict, max_dim)

# ***** Data Preparation for Training ******
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

CN_data = prepare_subject_data(subject_CN_dict, label=1)
AD_data = prepare_subject_data(subject_AD_dict, label=0)
all_data = CN_data + AD_data   # format is like this [[tensors of time_data], label], .... ]

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

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # Add pos encoding up to seq length
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes, num_heads=4, num_layers=4, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(model_dim, num_classes)

    def forward(self, x, mask):       
        B, T, _ = x.size()
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, D)

        # Update mask to account for cls token
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)  # (B, T+1)

        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        cls_out = x[:, 0, :]  # (B, D)

        out = self.dropout(cls_out)
        out = self.fc_out(out)  # (B, num_classes)
        return out

# train-test split
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
    all_labels = []
    all_probs = []
    
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

        # **** this is for AUC *****
        probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of positive class
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')

    return avg_loss, accuracy, auc

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

# training and testing
best_test_acc = 0
best_test_auc = 0
best_metrics = {}

epochs = 20
for epoch in range(epochs):
    train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, optimizer, criterion)
    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion)

    if (test_acc >= best_test_acc) and (test_auc >= best_test_auc):
        
        best_test_acc = test_acc
        best_test_auc = test_auc

        best_metrics = {
            'epoch': epoch + 1,
            'acc': test_acc,
            'auc': test_auc,
            'loss': test_loss
        }

    print(f"Epoch {epoch+1}/{epochs} - "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f} | "
          f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")

print(f"\nBest Test Accuracy: {best_metrics['acc']:.4f} "
      f"(Epoch {best_metrics['epoch']}) | AUC: {best_metrics['auc']:.4f}, "
      f"Loss: {best_metrics['loss']:.4f}")



