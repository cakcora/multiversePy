import networkx as nx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import json
import warnings
# from sklearn.tree import export_graphviz  # used in comment


def mineTrees(rf_model):
	"""
	:param rf_model:    trained random forest
	:return:            dataframe containing network metrics for each estimator in rf_model
	"""
	result = pd.DataFrame(index=np.arange(0, rf_model.n_estimators),
						  columns=['nodes', 'edges', 'diameter', 'weak_components', 'strong_components',
								   'node_connectivity', 'mean_hub_score', 'mean_auth_score',
								   'median_degree', 'mean_degree'])

	for t in range(0, rf_model.n_estimators):
		# print("Tree " + str(t) + " is processing")
		tree = rf_model.estimators_[t]
		graph = nx.DiGraph()  # Multiple edges are not allowed

		# export_graphviz(tree, out_file=str('results/trees/tree') + str(t) + '.dot',
		# feature_names=dataTrain.columns,class_names=data2.Class,rounded=True,
		# proportion=False,precision=2, filled=True)
		left_children = tree.tree_.children_left
		right_children = tree.tree_.children_right
		features = tree.tree_.feature

		for n in range(0, len(left_children)):
			node = features[n]

			l_child = left_children[n]
			r_child = right_children[n]

			if node >= 0:
				if l_child > 0 and features[l_child] >= 0:
					# print(str(t) + ">" + str(node) + " l" + str(l_child) + " " + str(features[l_child]))
					graph.add_edge(node, features[l_child])
				if r_child > 0 and features[r_child] >= 0:
					# print(str(t) + ">" + str(node) + " r" + str(r_child) + " " + str(features[r_child]))
					graph.add_edge(node, features[r_child])  # compare the graph with the original decision tree to make that the graph is correct

		# Network metrics
		with warnings.catch_warnings():  # temporarily suppress warnings
			warnings.filterwarnings('ignore', category=np.ComplexWarning)
			hubs, authorities = nx.hits_numpy(graph)

		mean_hub_score = np.mean(list(hubs.values()))  # hub = lots of links from
		mean_auth_score = np.mean(list(authorities.values()))  # authority = lots of links to

		nodes = nx.number_of_nodes(graph)
		if nodes == 0:  # empty tree would crash
			warnings.warn(f'Empty decision tree: t={t}', UserWarning)
			result.drop(index=t, inplace=True)  # data would be nan and crash next rf so delete row
			continue

		diameter = nx.diameter(nx.to_undirected(graph))  # greatest distance b/w any pair of vertices
		edges = nx.number_of_edges(graph)

		# size of subgraph where all components are connected
		strong_comp = nx.number_strongly_connected_components(graph)  # directed
		weak_comp = nx.number_weakly_connected_components(graph)  # ignore direction

		degrees = nx.average_degree_connectivity(graph, target="in")  # num incoming edges for vertices
		avg_in_degree = np.mean(list(degrees))
		median_in_degree = np.median(list(degrees))

		# This line is VERY slow, setting it to dummy value for now
		# node_connectivity = nx.average_node_connectivity(graph)  # how well the graph is connected
		node_connectivity = -1

		row = [nodes, edges, diameter, weak_comp, strong_comp,
			   node_connectivity, mean_hub_score, mean_auth_score,
			   median_in_degree, avg_in_degree]

		result.loc[t] = row
	return result


def poison(target_data, percentage, message=False):
	"""
	poison a percentage of a dataframe (target_data) and return a new dataframe

	:param target_data:     dataframe to poison
	:param percentage:      percentage of dataframe to poison
	:param message:         print a status message

	:return:                poisoned dataframe
	"""
	poisoned_data = target_data.copy()  # avoid poisoning original data

	if percentage > 0:
		length = len(poisoned_data.index)
		num_to_poison = int(percentage * length / 100)

		if message:
			print(f'out of {length} rows, labels of {num_to_poison} will be flipped')
		unique_vals = poisoned_data.Class.unique()

		if len(unique_vals) <= 1:
			print("ERROR: Dataset contains a single label for the Class feature")
		else:
			# don't poison from the beginning. pick i randomly and make sure you don't take same i two times
			for i in np.random.choice(poisoned_data.index.array, num_to_poison, replace=False):  # random indices
				new_label = poisoned_data.Class[i]

				while new_label == poisoned_data.Class[i]:
					new_label = np.random.choice(unique_vals)

				poisoned_data.at[i, 'Class'] = new_label  # fixes SettingWithCopyWarning

	return poisoned_data


def conf_matrix(prepped_data, major_max, minor_max, n_est=100, n_cv_folds=5, matrix_path=None):
	"""
	:param prepped_data:    encoded dataframe to train and test forests
	:param major_max:       number of major axis iterations
	:param minor_max:       number of minor axis iterations
	:param n_est:           n_estimators to use for random forest classifiers
	:param n_cv_folds:      number of folds for stratified k fold cross-validator
	:param matrix_path      path to folder for saving matrices (if None, does not save)
	"""

	entropy_list = []  # Will hold tuples of the form (Major, Entropy Value)

	# first test-train split
	y = prepped_data['Class']
	try:
		train, first_x_test, train['Class'], first_y_test = train_test_split(  # this recombines train for poisoning
			prepped_data.drop(columns=['Class']),
			y,
			random_state=1,
			stratify=y
		)
	except ValueError as e:
		warnings.warn(str(e), UserWarning)
		train, first_x_test, train['Class'], first_y_test = train_test_split(  # workaround for BTC Heist
			prepped_data.drop(columns=['Class']),
			y,
			random_state=1,
			stratify=None  # do not stratify
		)

	for major in range(0, major_max):
		data_poisoned_maj = poison(train, major)
		df = pd.DataFrame()

		for minor in range(0, minor_max):
			print("\nmajor:", major, " minor:", minor)
			data_poisoned_min = poison(data_poisoned_maj, minor, True)

			cv_results = cross_validate(
				RandomForestClassifier(n_estimators=n_est, random_state=42),
				data_poisoned_min.drop('Class', axis=1),  # train w/ poisoned
				data_poisoned_min['Class'],
				cv=n_cv_folds,  # num folds for default (StratifiedKFold)
				return_estimator=True
			)

			print(f'\tValidation accuracy:{cv_results["test_score"].mean():10.5f}')

			test_accuracies = []
			for trained_rf in cv_results['estimator']:
				# test on unseen data -> https://scikit-learn.org/stable/modules/cross_validation.html
				first_y_pred = trained_rf.predict(first_x_test)
				test_accuracies.append(accuracy_score(first_y_test, first_y_pred))

				# record estimators for second level
				result = mineTrees(trained_rf)
				result['minor'] = minor
				df = df.append(result)

			print(f'\tTest accuracy:{np.array(test_accuracies).mean():16.5f}')

		y = df['minor']
		x = df.drop('minor', axis=1)

		# Second test-train split
		x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, stratify=y)

		rf_second_level = RandomForestClassifier(n_estimators=n_est)
		rf_second_level.fit(x_train, y_train)

		y_pred_test = rf_second_level.predict(x_test)  # test second level model

		# The commented line is what was added during the meeting, however it was
		# throwing errors. This seems to be corrected now?
		# matrix = confusion_matrix(test_data.minor, y_pred_test)
		matrix = confusion_matrix(y_test, y_pred_test)
		plot_matrix(matrix, major, path=matrix_path)  # plot if path specified
		print(matrix)

		entropy = scipy.stats.entropy(matrix.flatten())
		entropy_list.append((major, entropy))

		if major == 0:
			first_matrix = matrix

	entropy_df = pd.DataFrame(entropy_list, columns=['Major', 'Entropy'])
	entropy_df.set_index('Major', inplace=True)
	entropy_df.to_csv(f'{config["out_csv_dir"]}{config["name"]}.csv')

	plot(entropy_df, matrix, first_matrix)


def plot(df_entropy, matrix, first_matrix):
	"""
	Plots the recorded entropy values and the final confusion matrix.
	:param df_entropy:      Entropy dataframe. Should contain index col of Major values, 'Entropy' column of entropy values.
	:param matrix:          Confusion matrix to plot
	"""
	sns.set(rc={'figure.figsize': (20, 10)})
	fig, ax = plt.subplots(1, 3)  # For 1 x 3 figures in plot

	ax[0] = sns.regplot(x=df_entropy.index, y=df_entropy['Entropy'], ax=ax[0])
	ax[0].set(title="Conf Matrix Entropy", xlabel="Major", ylabel="Entropy")

	ax[1] = sns.heatmap(first_matrix, annot=True, annot_kws={'size': 10},
						cmap=plt.cm.Greens, linewidths=0.2, ax=ax[2])  # show first matrix
	ax[1].set(title="First Confusion Matrix for Data Set " + config['name'])

	ax[2] = sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
						cmap=plt.cm.Greens, linewidths=0.2, ax=ax[1])  # show last matrix
	ax[2].set(title="Final Confusion Matrix for Data Set " + config['name'])

	plt.show()


def plot_matrix(matrix, major, path=None):
	"""
	plot and save confusion matrix
	:param matrix:      confusion matrix to plot
	:param major:       current major axis
	:param path:        path to folder for saving figure
	"""
	if path is not None:
		sns.set(rc={'figure.figsize': (7, 7)})

		ax = sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)
		ax.set(title=f'Confusion Matrix: {config["name"]}, major={major}')

		plt.savefig(path + f'conf_matrix_{config["name"]}_{major}.png')
		plt.close()  # garbage collect the figure


def entropy_plot(configs):
	"""
	make plot to compare entropy between datasets
	:paramgi configs:     list of dataset configurations
	"""
	sns.set(rc={'figure.figsize': (20, 10)})
	fig, ax = plt.subplots()

	for c in configs:
		df = pd.read_csv(f'{c["out_csv_dir"]}{c["name"]}.csv')
		ax.plot(df['Major'], df['Entropy'], label=c['name'])

	ax.set_title('Entropy Plot')
	ax.set_xlabel('Major')
	ax.set_ylabel('Entropy')
	ax.legend()

	plt.savefig(f'{c["out_csv_dir"]}Dataset Plot.png')
	plt.close()


def prep_data(conf):
	"""
	read and preprocess data according to settings in conf
	:param conf:        dictionary of configurations
	:return:            dataframe where x is one-hot encoded and class is categorically encoded
	"""
	if conf['ignore_head']:
		skip_row = 1
	else:
		skip_row = 0

	raw = pd.read_csv(conf['file_path'], names=conf['column_names'], skiprows=skip_row, header=None)

	# If the sample size is set to 0, just use the entire data set
	# Otherwise, draw sample
	if conf['sample_size'] > 0:
		raw = raw.sample(n=conf['sample_size'])

	# one-hot encoding of the raw (except for the Class variable and ignored columns)
	encoded = pd.get_dummies(raw.drop(columns=['Class'] + conf['ordinal_encode_columns']))

	# ordinal encode and add back ignored columns (currently just BTC address)
	for col_name in conf['ordinal_encode_columns']:
		encoded[col_name] = pd.factorize(raw[col_name])[0]  # codes, not unique values

	le = preprocessing.LabelEncoder()  # encode Class variable numerically
	encoded['Class'] = le.fit_transform(raw['Class'])

	# reset major_max in config if specified
	if conf['major_max'] is None or conf['major_max'] >= (100 / len(le.classes_)):
		conf['major_max'] = (100 // len(le.classes_)) - 1  # reset major to max possible

	return encoded


def get_configs():
	"""
	gets and consolidates configs for each dataset
	:return:    list of config dictionaries
	"""
	config_file = "../config/config.json"  # relative path to config file
	with open(config_file, 'rt') as f:
		config_full = json.load(f)

	global_conf = config_full['global']
	datasets = config_full['dataset']
	default = datasets['default']  # default configs for datasets

	# config dictionaries for each dataset: conf comes after default so it will replace duplicate keys
	configs = [{'name': name, **global_conf, **default, **conf} for name, conf in datasets.items() if name != 'default']
	return configs


if __name__ == '__main__':
	# Each data set is an element in the configs list
	# Loop through and process each.
	configs = get_configs()

	for config in configs:
		print("PROCESSING DATA SET: " + config['name'])
		data = prep_data(config)
		conf_matrix(data, config['major_max'], config['minor_max'], n_est=config['n_estimators'],
					n_cv_folds=config['n_cv_folds'], matrix_path=f'{config["graph_dir"]}{config["name"]}/')

	entropy_plot(configs)

"""
***************************************************************************************************************
* Note: The bitcoin data set is >200mb, it cannot be stored on github, you will have to download it yourself  *
*       and put the csv in data/bitcoin_heist/BitcoinHeistData.csv                                            *
***************************************************************************************************************

TODO:
1. Mess around with 20x20 conf matrix
2. Check sci kit library for cross validation and random forest 
	3. Add global configs
4. AUC & log loss & bias
5. Scaling presentation and auto UCI data downloading, 
6. crossvalidation (Steven)
7. Class imbalance & oversampling/upsampling/downsampling (smote)
	8. Cap / autogenerate major_max
	9. Entropy vs major graph for all datasets

Shapley values for feature importance (which features are important) https://dalex.drwhy.ai/
(book https://ema.drwhy.ai/shapley.html)

========================================================================================================================

DATASETS:
adult:		https://archive.ics.uci.edu/ml/datasets/Adult
bc:			https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
btc heist:	https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset

========================================================================================================================

ISSUES:

1)
- Fix: skip decision trees where the graph is empty, (higher sample size should lower issue frequency).
- Caused by an empty decision tree (no decision cause it had one class for training).
- Or an empty decision tree graph (idk how these arent the same thing but it still crashes if i check tree node count).
- Happens rarely with Adult and Breast Cancer ds, often with BTC Heist set.

D:\Code\poison\venv\lib\site-packages\numpy\core\fromnumeric.py:3420: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
D:\Code\poison\venv\lib\site-packages\numpy\core\_methods.py:188: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
Traceback (most recent call last):
  File "D:/Code/poison/multiversePy/src/main.py", line 221, in <module>
    conf_matrix(data_train, data_test, config['major_max'], config['minor_max'], n_est=config['n_estimators'])
  File "D:/Code/poison/multiversePy/src/main.py", line 138, in conf_matrix
    result = mineTrees(rf_first_level)
  File "D:/Code/poison/multiversePy/src/main.py", line 58, in mineTrees
    diameter = nx.diameter(nx.to_undirected(graph))  # greatest distance b/w any pair of vertices
  File "D:\Code\poison\venv\lib\site-packages\networkx\algorithms\distance_measures.py", line 300, in diameter
    return max(e.values())
ValueError: max() arg is an empty sequence

2)
- Fix: workaround = try-catch in conf_matrix for train_test_split (higher sample size should lower issue frequency).
- A better fix might be to take a stratified sample in pred_data, this should solve the root issue.
- It throws a ValueError for BTC Heist dataset because it cannot stratify.
- It cannot stratify because there too few examples of the minority classes in the sample.

Traceback (most recent call last):
  File "/home/jon/PycharmProjects/multiversePy/src/main.py", line 250, in <module>
    conf_matrix(data, config['major_max'], config['minor_max'], n_est=config['n_estimators'])
  File "/home/jon/PycharmProjects/multiversePy/src/main.py", line 132, in conf_matrix
    train, first_x_test, train['Class'], first_y_test = train_test_split(  # recombine train x and y for poisoning
  File "/home/jon/PycharmProjects/multiversePy/venv/lib/python3.9/site-packages/sklearn/model_selection/_split.py", line 2197, in train_test_split
    train, test = next(cv.split(X=arrays[0], y=stratify))
  File "/home/jon/PycharmProjects/multiversePy/venv/lib/python3.9/site-packages/sklearn/model_selection/_split.py", line 1387, in split
    for train, test in self._iter_indices(X, y, groups):
  File "/home/jon/PycharmProjects/multiversePy/venv/lib/python3.9/site-packages/sklearn/model_selection/_split.py", line 1715, in _iter_indices
    raise ValueError("The least populated class in y has only 1"
ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.

"""
