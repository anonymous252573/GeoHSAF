import torch
import gpytorch
import os
import pickle
import numpy as np
import GPy
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
            return None
    
    valid_labels = [label for label in time_labels if convert(label) is not None]
    time_map = {label: convert(label) for label in sorted(valid_labels, key=convert)}
    return time_map    #bl = 1, m03 = 3, m06= 6, m12 = 12 , etc; returns {'bl': 1, 'm03': 3, ....}  

def load_pkl_data(folder, time_map):
    all_X, all_Y = [], []
    subject_data = {}

    subject_id_map = {}   
    current_id = 1

    for file in tqdm(os.listdir(folder)):
        if not file.endswith('.pkl'):
            continue

        filepath = os.path.join(folder, file)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        subject_str_id = file.split('_pca')[0]

        # Assign an integer ID if not already done
        if subject_str_id not in subject_id_map:
            subject_id_map[subject_str_id] = current_id
            current_id += 1

        subject_int_id = subject_id_map[subject_str_id]
        subject_data[subject_str_id] = {}

        for time_key, vector in data.items():
            if time_key not in time_map:
                continue
            t = time_map[time_key]
            all_X.append([subject_int_id, t])  # Append time and subject ID together
            all_Y.append(vector)
            subject_data[subject_str_id][t] = vector

    return np.array(all_X), np.vstack(all_Y), subject_data, subject_id_map   #subject data (ID and the PCA vector across time); subject_id_map (encode the ID)


condis = ['AD', 'CN', 'MCI']
for condi in condis:
    data_folder = r'C:\.....\AIBL\After_matching\{}_PCA_tangent_data'.format(condi) 
    save_folder = r'C:\......\AIBL\After_matching\{}_interpolated_PCA_tangent_data'.format(condi)
    os.makedirs(save_folder, exist_ok=True) 

    time_map = extract_time_map(data_folder)  # e.g., {'bl': 1, 'm03': 3, 'm06': 6, 'm12': 12}
    X, Y, subject_data, subject_id_maps = load_pkl_data(data_folder, time_map)
    # print(np.linalg.norm(Y, axis=-1))  - valid

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

    # *** Shuffling ****
    indices = torch.randperm(X_tensor.size(0))
    X_tensor = X_tensor[indices]
    Y_tensor = Y_tensor[indices]

    train_x, val_x, train_y, val_y = train_test_split(
        X_tensor, Y_tensor, test_size=0.2, random_state=42
    )

    print('Train data size: ', train_x.shape, train_y.shape)
    print('Val data size: ', val_x.shape, val_y.shape)
    # print(torch.norm(train_y, dim=-1))  - valid

    # GPR
    num_latents = 3
    num_tasks = Y_tensor.size(1)
    class MultitaskGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, num_latents, num_tasks):
            inducing_points = torch.rand(num_latents, 16, 2)

            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_latents])
            )

            variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                ),
                num_tasks=num_tasks,
                num_latents=num_latents,
                latent_dim=-1
            )

            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents]), ard_num_dims=2),
                batch_shape=torch.Size([num_latents])
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
    model = MultitaskGPModel(num_latents, num_tasks).to(device)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.1)

    num_epochs = 35
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
    nll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=val_y.size(0))
    train_losses = []
    val_losses = []
    train_stds = []
    val_stds = []

    epochs_iter = tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        model.train()
        likelihood.train()

        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        epochs_iter.set_postfix(loss=loss.item())
        print(f'\nTrain Loss {loss}')

        # Evaluate on training and validation sets every 5 epochs
        if (i + 1) % 5 == 0:
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                # Predict on training set
                train_pred = likelihood(model(train_x))
                train_loss = -mll(model(train_x), train_y).item()
                train_mean = train_pred.mean.mean().item()
                train_std = train_pred.stddev.mean().item()

                # Predict on validation set
                val_pred = likelihood(model(val_x))
                # print(train_pred.mean.shape, val_pred.mean.shape)
                val_loss = -nll(model(val_x), val_y).item()
                val_mean = val_pred.mean.mean().item()
                val_std = val_pred.stddev.mean().item()

                # Save metrics
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_stds.append(train_std)
                val_stds.append(val_std)

            print(f"\nEpoch {i+1} | "
                f"Train Loss: {train_loss:.4f}, Mean: {train_mean:.4f}, Stddev: {train_std:.4f} | "
                f"Val Loss: {val_loss:.4f}, Mean: {val_mean:.4f}, Stddev: {val_std:.4f}")

    epochs = list(range(5, (len(train_losses) + 1) * 5, 5))
    # Plot 1: Losses
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("ELBO Loss")
    plt.title("Train vs. Validation Loss")
    plt.legend()
    plt.grid(True)

    # Plot 2: Standard Deviations
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_stds, label='Train Std', marker='x')
    plt.plot(epochs, val_stds, label='Val Std', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Predictive Std Dev")
    plt.title("Train vs. Validation Std Dev")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
            
    # Interpolation
    inverse_time_map = {v: k for k, v in time_map.items()}
    new_entries = {9: 'm09', 14: 'm14', 27: 'm27', 45: 'm45', 63: 'm63'}
    inverse_time_map.update(new_entries)

    for subject_key, time_dict in subject_data.items():  
        subject_int_id = subject_id_maps.get(subject_key)  
        if subject_int_id is None:
            print(f'Cannot find the ID for {subject_key}')

        existing_time_points = set(time_dict.keys())
        full_time_points = set(time_map.values())
        add_time = {9, 14, 27, 45, 63}
        full_time_points.update(add_time)

        missing_time_points = full_time_points - existing_time_points
        full_time_data = {}
        for time_int, data_array in time_dict.items():
            time_str = inverse_time_map[time_int]
            full_time_data[time_str] = data_array

        # Fill missing time points
        for missing_time_int in missing_time_points:
            time_str = inverse_time_map[missing_time_int]
            model.eval()
            likelihood.eval()
            
            with torch.no_grad():
                input_val = torch.tensor(
                    np.array([[subject_int_id, missing_time_int]]),
                    dtype=torch.float32
                ).to(device)

                filled_array = likelihood(model(input_val))
                filled_array = filled_array.mean
                full_time_data[time_str] = filled_array.cpu().numpy()

        # Save the completed dictionary as a .pkl file
        filename = f"{subject_key}_interpolated.pkl"
        save_path = os.path.join(save_folder, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(full_time_data, f)

        print(f"Saved interpolated data for {subject_key} to {save_path}")