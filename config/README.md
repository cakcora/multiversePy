# Config Description

* name - The name of the dataset
* class_names - File containing dataset description, currently not used
* census_file - The actual dataset
* ignore_head - If the dataset contains labels as the first row, set to true
* entropy_csv - Path of csv that will be written after data set is processed
* ordinal_encode_columns - Array of column names to encode ordinal (not one-hot)
* class_column - Column number of the data set labels
* column_names - Array of column names
* major_max - Base level of poisoning on the first level rf
* minor_max - Additional poisoning on second level rf
* sample_size - Set to 0 to simply take entire dataset, otherwise set to sample size.
* n_estimators - Number of estimators used in random forest model
* test_fraction - [0, 100] Percentage of dataset used for test data 