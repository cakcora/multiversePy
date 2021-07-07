# Config Description

## Global Configs

These configs apply to all datasets.

* major_max - Base level of poisoning on the first level rf (set to null to use max possible: (100/num labels) - 1)
* minor_max - Additional poisoning on second level rf
* n_cv_folds - Number of folds for cross validation of inner loop
* n_estimators - Number of estimators used in random forest model
* param_grid - Parameters for [hyperparameter](https://scikit-learn.org/stable/modules/ensemble.html#random-forest-parameters) optimization of random forests
* test_fraction - \[0, 100\] Percentage of dataset used for test data 
* out_csv_dir - Folder for output csv files
* graph_dir - Folder for saved output graphs

## Dataset Configs

Specify settings for each dataset. There are default options that can be overridden by defining them for the individual dataset.

* sample_size - Set to 0 to simply take entire dataset, otherwise set to sample size
* file_path - The actual dataset
* ignore_head - If the dataset contains labels as the first row, set to true
* ordinal_encode_columns - Array of column names to encode ordinal (not one-hot)
* class_column - Column number of the data set labels
* column_names - Array of column names
