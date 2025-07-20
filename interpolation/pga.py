import os
import pickle
import numpy as np
from glob import glob
from geomstats.geometry.pre_shape import PreShapeSpace
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

n_components_list = [round(x, 2) for x in np.arange(0.1, 1.0, 0.05)]

condis = ['AD', 'CN', 'MCI']
for condi in condis:

    m, k = 732, 3 
    shape_space = PreShapeSpace(m, k)
    karcher_mean_path = r"C:\.....\After_matching\{}_Karcher_mean\{}_Karcher_mean_shape.pkl".format(condi, condi)
    with open(karcher_mean_path, 'rb') as f:
        mean_shape_dict = pickle.load(f)

    def process_reconstructed_value(value, mean_shape):
        return shape_space.metric.exp(value, mean_shape)

    def compute_mse(reconstructed_data, true_data):
        return np.mean((reconstructed_data - true_data) ** 2)
    
    def compute_mae(reconstructed_data, true_data):
        return np.mean(np.abs(reconstructed_data - true_data))

    def pca_reconstruction(folder_a, folder_b, save_folder, save_best_pca_model_folder):

        files_b = sorted([f for f in os.listdir(folder_b) if f.endswith('.pkl')])

        pca_models = []
        explained_variances = []
        mse_train_list = []
        mse_test_list = []

        # Step 2: Collect data from tangent folder for PCA
        all_data_b = []
        metadata_b = []  # List to store (file_name, time_key) tuples for each sample
        for file in files_b:
            with open(os.path.join(folder_b, file), 'rb') as f:
                data_b = pickle.load(f)
                for time_key, data in data_b.items():
                    all_data_b.append(data.flatten())  
                    metadata_b.append((file, time_key))  # Store the (file_name, time_key) pair

        # Stack 
        all_data_b = np.stack(all_data_b, axis=0) 

        # Split data into train and test
        X_train, X_test, metadata_train, metadata_test = train_test_split(
            all_data_b, metadata_b, test_size=0.2, random_state=42
        )

        for n_components in n_components_list: 
            pca = PCA(n_components=n_components)
            pca.fit(X_train)

            # Transform both train and test data
            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test)

            # Inverse transform
            X_train_full = pca.inverse_transform(X_train_pca)
            X_test_full = pca.inverse_transform(X_test_pca)

            # exponential map at the mean
            X_train_reconstructed = np.array([
                process_reconstructed_value(
                    data.reshape(m, k),                
                    mean_shape_dict[time_point]
                )
                for data, (_, time_point) in zip(X_train_full, metadata_train)
            ])

            X_test_reconstructed = np.array([
                process_reconstructed_value(
                    data.reshape(m, k),
                    mean_shape_dict[time_point]
                )
                for data, (_, time_point) in zip(X_test_full, metadata_test)
            ])

            print(X_train_reconstructed.shape, X_test_reconstructed.shape)

            # MAE for train and test
            mse_train = []
            mse_test = []

            for i, (file_name, time_key) in enumerate(metadata_train):
                with open(os.path.join(folder_a, file_name), 'rb') as f:
                    data_train = pickle.load(f)
                    true_data = data_train[time_key] 
                    reconstructed_data = X_train_reconstructed[i] 
                    # mse_train.append(compute_mse(reconstructed_data, true_data))
                    mse_train.append(compute_mae(reconstructed_data, true_data))

            for i, (file_name, time_key) in enumerate(metadata_test):
                with open(os.path.join(folder_a, file_name), 'rb') as f:
                    data_test = pickle.load(f)
                    true_data = data_test[time_key] 
                    reconstructed_data = X_test_reconstructed[i]
                    # mse_test.append(compute_mse(reconstructed_data, true_data))
                    mse_test.append(compute_mae(reconstructed_data, true_data))

            # Save model and errors
            pca_models.append(pca)
            explained_variances.append(np.sum(pca.explained_variance_ratio_))
            mse_train_list.append(np.mean(mse_train))
            mse_test_list.append(np.mean(mse_test))

            print(f"Components: {pca.n_components_} | Train MSE: {np.mean(mse_train)} | Test MSE: {np.mean(mse_test)}")

        # best model based on the error
        best_pca_idx = np.argmin(mse_test_list)
        best_pca = pca_models[best_pca_idx]
        best_mse = mse_test_list[best_pca_idx]

        # save model
        pca_dict = {'pca_model': best_pca}
        save_path  = os.path.join(save_best_pca_model_folder, f'{condi}_pca_model.pkl')
        with open(save_path, 'wb') as f:
                pickle.dump(pca_dict, f)

        for filename in os.listdir(folder_b):
            if filename.endswith('.pkl'):
                filepath = os.path.join(folder_b, filename)
                with open(filepath, 'rb') as f:
                    data_dict = pickle.load(f)

            new_dict = {} 
            for key, array in data_dict.items():
                flat_array = array.reshape(-1)
                transformed = best_pca.transform(flat_array.reshape(1, -1))[0]
                new_dict[key] = transformed
        
            base_name = os.path.splitext(filename)[0]  # removes .pkl extension
            save_name = f"{base_name}_pca_result.pkl"
            save_path = os.path.join(save_folder, save_name)
            with open(save_path, 'wb') as f:
                pickle.dump(new_dict, f)

        print("All PGA-transformed files saved.")
        
        # Plot the explained variance and MSE for train and test samples
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(explained_variances, mse_train_list, marker='o', label="Train Error")
        plt.plot(explained_variances, mse_test_list, marker='s', label="Test Error")
        # plt.title("PGA Variance vs Reconstruction Error")
        plt.xlabel("PGA Explained Variance")
        plt.ylabel("Mean Reconstruction Error")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("pga_vs_error.png")
        plt.xticks(np.arange(min(explained_variances), max(explained_variances) + 0.05, 0.1))
        plt.show()
        

    original_folder = r"C:\.....\ADNI\After_matching\{}_orig_shape_space".format(condi)
    tangent_folder = r'C:\.....\ADNI\After_matching\{}_tangent_data'.format(condi)
    output_folder = r'C:\.....\ADNI\After_matching\{}_PGA_tangent_data'.format(condi)
    os.makedirs(output_folder, exist_ok=True)
    save_pca_folder = r'C:\.....\ADNI\After_matching\{}_PGA_model'.format(condi)
    os.makedirs(save_pca_folder, exist_ok=True)

    pca_reconstruction(original_folder, tangent_folder, output_folder, save_pca_folder)