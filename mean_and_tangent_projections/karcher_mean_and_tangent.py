import os
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
import matplotlib.pyplot as plt
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.geometry.pre_shape import KendallShapeMetric
from geomstats.learning.frechet_mean import FrechetMean
import geomstats.backend as gs
from collections import defaultdict
import pickle


condis = ['AD', 'CN', 'MCI']
for condi in condis:
    save_folder_orig = r'C:\....\After_matching\{}_orig_shape_space'.format(condi)
    os.makedirs(save_folder_orig, exist_ok=True)
    save_folder_tangent_data = r'C:\....\After_matching\{}_tangent_data'.format(condi)
    os.makedirs(save_folder_tangent_data, exist_ok=True)
    save_folder_karcher = r'C:\....\After_matching\{}_Karcher_mean'.format(condi)
    os.makedirs(save_folder_karcher, exist_ok=True)

    folder_path = r'C:\....\ADNI\After_matching\{}'.format(condi)

    # Dictionary to hold subject_id as key and list of timepoints as value
    subject_timepoints = defaultdict(list)

    # Set to hold unique timepoints
    unique_timepoints = set()

    # Counter for total number of processed files
    total_scans = 0

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') and filename.startswith('matched_'):
            total_scans += 1  # Count the scan

            # Remove 'matched_' prefix and '.txt' suffix
            base = filename[len('matched_'):].replace('.txt', '')

            # Split subject ID and timepoint
            if '_' in base:
                *subject_parts, timepoint = base.split('_')
                subject_id = '_'.join(subject_parts)

                subject_timepoints[subject_id].append(timepoint)
                unique_timepoints.add(timepoint)

    # Count subjects without a 'bl' timepoint
    subjects_without_bl = [sid for sid, times in subject_timepoints.items() if 'bl' not in times]
    subjects_with_bl = [sid for sid, times in subject_timepoints.items() if 'bl' in times]
    total_scans_with_bl = sum(len(times) for sid, times in subject_timepoints.items() if 'bl' in times)

    print(f"Total number of scans (processed files): {total_scans}")
    print(f"Total number of subjects: {len(subject_timepoints.keys())}")
    print(f"Total number of subjects without 'bl': {len(subjects_without_bl)}\n")

    print(f"Total number of scans with bl: {total_scans_with_bl}")

    print("\nUnique Timepoints:")
    print(f'{len(unique_timepoints)} Time points : {sorted(unique_timepoints)}')

    m, k = 732, 3
    shape_space = PreShapeSpace(m, k)
    def frobenius_norm(matrix):
        return np.linalg.norm(matrix, ord='fro')
    
    temporal_shapes = defaultdict(list)

    # Load and group shapes by timepoint
    for subject_id in subjects_with_bl:
        timepoints = subject_timepoints[subject_id]

        for timepoint in timepoints:
            filename = f"matched_{subject_id}_{timepoint}.txt"
            full_path = os.path.join(folder_path, filename)

            after_vectors = []
            with open(full_path, 'r') as f:
                for line in f:
                    _, after = line.split('->')
                    vector_after = np.array([float(x) for x in after.strip().split()])
                    after_vectors.append(vector_after)

            after_matrix = np.array(after_vectors)
            norm_after = frobenius_norm(after_matrix)
            normalized_after = after_matrix / norm_after

            temporal_shapes[timepoint].append(normalized_after)

    # Compute mean shape for each timepoint
    mean_shape_dict = {}
    total_shapes_count = 0

    for timepoint, shape_list in temporal_shapes.items():
        shape_array = np.transpose(np.stack(shape_list, axis=-1), (2, 0, 1))
        total_shapes_count += shape_array.shape[0]

        print(np.min(np.array(shape_list)), np.max(np.array(shape_list)))

        print(f"Computing Karcher mean for timepoint {timepoint}, with {shape_array.shape[0]} shapes...")

        shape_array = gs.array(shape_array)
        frechet_mean = FrechetMean(space=shape_space)
        mean_shape_ = frechet_mean.fit(shape_array).estimate_

        tan_mean_shape = shape_space.metric.log(mean_shape_, mean_shape_)

        mean_shape_dict[timepoint] = mean_shape_

    save_filename_mean = f"{condi}_Karcher_mean_shape.pkl"
    save_path = os.path.join(save_folder_karcher, save_filename_mean)
    with open(save_path, 'wb') as f:
        pickle.dump(mean_shape_dict, f)

    print(f"Total number of shapes across all timepoints: {total_shapes_count}")

    # tangent projections
    for subject_id in subjects_with_bl:
        timepoints = subject_timepoints[subject_id]

        orig_shape_dict = {}
        tangent_dict = {}

        for timepoint in timepoints:
            filename = f"matched_{subject_id}_{timepoint}.txt"
            full_path = os.path.join(folder_path, filename)

            after_vectors = []
            with open(full_path, 'r') as f:
                for line in f:
                    # Split the line at the '->' symbol
                    before, after = line.split('->')
                    
                    vector_after = np.array([float(x) for x in after.strip().split()])
                    after_vectors.append(vector_after)

            after_matrix = np.array(after_vectors)
            norm_after = frobenius_norm(after_matrix)
            normalized_after = after_matrix / norm_after
            mean_shape = mean_shape_dict[timepoint]
            normalized_after_tan = shape_space.metric.log(normalized_after, mean_shape)

            orig_shape_dict[timepoint] = normalized_after
            tangent_dict[timepoint] = normalized_after_tan
            
        # save
        timepoint_str = '_'.join(sorted(str(k) for k in orig_shape_dict.keys()))
        save_filename = f"{subject_id}_{timepoint_str}.pkl"

        save_path_orig = os.path.join(save_folder_orig, save_filename)
        save_path_tan = os.path.join(save_folder_tangent_data, save_filename)

        with open(save_path_orig, 'wb') as f:
            pickle.dump(orig_shape_dict, f)

        with open(save_path_tan, 'wb') as f:
            pickle.dump(tangent_dict, f)

    print("All completed")




        






