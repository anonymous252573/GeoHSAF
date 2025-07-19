## **GeoHSAF: Geometric Hippocampus Shape Analysis Framework for Longitudinal Alzheimer's Disease Classification**

# Dataset and Codes

### Dataset
We use three public longitudinal AD datasets: ADNI, OASIS and AIBL. In the 'Datasets' folder, we include the scripts to download and organize the MR images after downloading. The scripts should be run in the following order:
- #### ADNI
  **⚡(1)** Run 'get_category_to_csv.py' file to extract subjects as .csv  **⚡(2)** Run 'get_ptid_vscode_and_uid.py' file to get the subjects ID for downloading the MRI scans from LONI **⚡(3)** After downloading the MRI scans, run 'convert_dcm_to_nii.py' to convert dicom files to .nii **⚡(4)** Run 'count_nifti.py' to ensure all files are correctly converted **⚡(5)** Run 'preprocessed_to_timepoints.py' to organize files for segmentation.
- #### OASIS
  Download OASIS-2 from here]([https://sites.wustl.edu/oasisbrains/datasets/]). Then Run **⚡(1)** the file 'convert_files_to_nii.py' to convert files to .nii and **⚡(2)** the file 'extract_time_points.py' to organize for segmentation.
- #### AIBL
  


