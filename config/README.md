# Config Description

## Run Configs

These configs apply to all datasets.

* major_max - Base level of poisoning on the first level rf (set to null to use max possible: (100/num labels) - 1)
* minor_max - Additional poisoning on second level rf
* n_cv_folds - Number of folds for cross validation of inner loop
* n_estimators - Number of estimators used in random forest model
* sample_size - Set to 0 to simply take entire dataset, otherwise set to sample size
* param_grid - Parameters for hyperparameter optimization of random forests
* test_fraction - \[0, 100\] Percentage of dataset used for test data 
* out_csv_dir - Folder for output csv files
* temp_csv_dir - Folder for temporary csv files
* preprocess_dir - Folder for preprocessed csv files
* matrix_dir - Folder for saved confusion matrices
* datasets - List of dataset csv files in preprocess_dir folder

## Preprocess Configs

Specify settings for each dataset. There are default options that can be overridden by defining them for the individual dataset.
* out_dir - Folder to store preprocessed datasets
* data_path - Input dataset
* ordinal_encode_columns - Array of column numbers to encode ordinal (one-hot is default)
* drop_columns - Array of column numbers to drop
* datetime_columns - Array of column numbers to convert to datetime
* class_column - Column number of the dataset labels
