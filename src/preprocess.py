import pandas as pd
import numpy as np
from sklearn import preprocessing
import json
import os
from time import perf_counter


CONFIG_FILE = '../config/preprocessing_configs.json'


def preprocess_dataset(conf):
	"""
	read and preprocess data according to settings in conf
	:param conf:        dictionary of configurations
	"""
	print(f'Dataset: {conf["filename"]}')

	skip_row = 1 if conf['ignore_head'] else 0  # skip if first row is not data
	raw = pd.read_csv(conf['data_path'], skiprows=skip_row, header=None)

	for col in conf['datetime_columns']:
		raw[col] = pd.to_datetime(raw[col]).astype(int) // 10**9  # convert to seconds

	# replace NaN with median
	for col in raw.columns:  # finance and Rain in Australia datasets are full of nan
		if np.issubdtype(raw[col].dtype, np.number):
			raw[col].replace(np.NaN, raw[col].mean(), inplace=True)

	# if class column is set to -1, use last column
	class_col = conf['class_column'] if conf['class_column'] != -1 else len(raw.columns) - 1
	raw.rename(columns={class_col: 'Class'}, inplace=True)

	# one-hot encoding of the raw (except for the Class variable and ordinal-encoded/ignored columns)
	encoded = pd.get_dummies(raw.drop(columns=['Class'] + conf['ordinal_encode_columns'] + conf['drop_columns']))

	# ordinal encode and add back ordinal encoded columns
	for col_name in conf['ordinal_encode_columns']:
		encoded[col_name] = pd.factorize(raw[col_name])[0]  # codes, not unique values

	le = preprocessing.LabelEncoder()  # encode Class variable numerically
	encoded['Class'] = le.fit_transform(raw['Class'])

	encoded.to_csv(f'{conf["out_dir"]}{conf["filename"]}.csv', index=False, header=False)


def get_configs():
	"""
	gets and consolidates configs for each dataset
	:return:    list of config dictionaries
	"""
	with open(CONFIG_FILE, 'rt') as f:
		config_full = json.load(f)

	datasets = config_full['datasets']
	default = config_full['default']  # default configs for datasets

	# config dictionaries for each dataset: conf comes after default so it will replace duplicate keys
	configs = [{'name': name,  **default, **conf} for name, conf in datasets.items()]
	for c in configs:
		c['filename'] = c['name'].replace(' ', '_').lower()  # clean filename
	return configs


def preprocess(dataset_names=None):
	"""
	preprocess datasets in configuration file
	:param dataset_names:   names of specific datasets to preprocess, if None -> all datasets
	"""
	os.chdir(os.path.dirname(os.path.realpath(__file__)))  # make sure we are in multiversePy/src/
	print('Preprocessing Datasets')

	configs = get_configs()
	if not os.path.exists(out_dir := configs[0]['out_dir']):
		os.mkdir(out_dir)

	configs = [config for config in configs if dataset_names is None or config['name'] in dataset_names]
	for config in configs:
		preprocess_dataset(config)


if __name__ == '__main__':
	t = perf_counter()
	preprocess(dataset_names=['Bitcoin Heist', 'kdd cup'])
	print(f'Time: {perf_counter() - t:.2f}')
