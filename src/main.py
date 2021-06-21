import random
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import scipy.stats
import json
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
		hubs, authorities = nx.hits_numpy(graph)
		mean_hub_score = np.mean(list(hubs.values()))  # hub = lots of links from
		mean_auth_score = np.mean(list(authorities.values()))  # authority = lots of links to

		nodes = nx.number_of_nodes(graph)
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
					new_label = random.choice(unique_vals)

				poisoned_data.at[i, 'Class'] = new_label  # fixes SettingWithCopyWarning

	return poisoned_data


def conf_matrix(train_data, test_data, major_max, minor_max, n_est=100):
	"""
	:param train_data:      dataframe to train forests
	:param test_data:       dataframe to test forests
	:param major_max:       number of major axis iterations
	:param minor_max:       number of minor axis iterations
	:param n_est:           n_estimators to use for random forest classifiers
	"""

	entropy_list = []  # Will hold tuples of the form (Major, Entropy Value)

	for major in range(0, major_max):
		data_poisoned_maj = poison(train_data, major)
		df = pd.DataFrame()

		for minor in range(0, minor_max):
			data_poisoned_min = poison(data_poisoned_maj, minor, True)
			print("\tmajor:", major, " minor:", minor)

			rf_first_level = RandomForestClassifier(n_estimators=n_est, random_state=42)
			rf_first_level.fit(data_poisoned_min.drop('Class', axis=1), data_poisoned_min['Class'])  # train w/ poisoned2

			result = mineTrees(rf_first_level)
			result['minor'] = minor
			df = df.append(result)

		y = df['minor']
		x = df.drop('minor', axis=1)

		# Second test-train split
		x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, stratify=y)

		rf_second_level = RandomForestClassifier(n_estimators=n_est)
		rf_second_level.fit(x_train, y_train)

		y_pred_test = rf_second_level.predict(x_test)  # test model against un-poisoned data

		# The commented line is what was added during the meeting, however it was
		# throwing errors. This seems to be corrected now?
		# matrix = confusion_matrix(test_data.minor, y_pred_test)
		matrix = confusion_matrix(y_test, y_pred_test)
		print(matrix)

		entropy = scipy.stats.entropy(matrix.flatten())
		entropy_list.append((major, entropy))

		if major == 0:
			first_matrix = matrix

	entropy_df = pd.DataFrame(entropy_list, columns=['Major', 'Entropy'])
	entropy_df.set_index('Major', inplace=True)
	entropy_df.to_csv(config["entropy_csv"])

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

	ax[1] = sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
						cmap=plt.cm.Greens, linewidths=0.2, ax=ax[1])  # show last matrix
	ax[1].set(title="Final Confusion Matrix for Data Set " + config['name'])

	ax[2] = sns.heatmap(first_matrix, annot=True, annot_kws={'size': 10},
						cmap=plt.cm.Greens, linewidths=0.2, ax=ax[2])  # show first matrix
	ax[2].set(title="First Confusion Matrix for Data Set " + config['name'])

	plt.show()


def read_sample_split_data(conf):
	"""
	read and preprocess data according to settings in conf
	:param conf:        dictionary of configurations
	:return:            tuple of dataframes: (training-data, testing-data)
	"""
	if conf['ignore_head']:
		skip_row = 1
	else:
		skip_row = 0

	raw = pd.read_csv(conf['census_file'], names=conf['column_names'], skiprows=skip_row, header=None)

	# If the sample size is set to 0, just use the entire data set
	# Otherwise, draw sample
	if conf['sample_size'] > 0:
		raw = raw.sample(n=conf['sample_size'])

	# one-hot encoding of the raw (except for the Class variable)
	train = pd.get_dummies(raw.loc[:, raw.columns != 'Class'])

	le = preprocessing.LabelEncoder()  # encode Class variable numerically
	train['Class'] = le.fit_transform(raw['Class'])

	# TODO: Need a second train test set for rf_second_level
	# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

	test = train.sample(frac=conf['test_fraction'])
	train.drop(index=train.index.intersection(test.index), inplace=True)  # remove rows that are in test

	return train, test


if __name__ == '__main__':
	config_file = "../config/config.json"  # relative path to config file
	with open(config_file, 'rt') as f:
		config_full = json.load(f)

	# Each data set is an element in the config array.
	# Loop through and process each.
	for i in range(0, len(config_full)):
		config = config_full[i]
		print("PROCESSING DATA SET: " + config['name'])
		data_train, data_test = read_sample_split_data(config)
		conf_matrix(data_train, data_test, config['major_max'], config['minor_max'], n_est=config['n_estimators'])

"""
***************************************************************************************************************
* Note: The bitcoin data set is >200mb, it cannot be stored on github, you will have to download it yourself  *
*       and put the csv in data/bitcoin_heist/BitcoinHeistData.csv                                            *
***************************************************************************************************************

TODO:
1. 2 test_train splits for rf levels
2. Mess around with 20x20 conf matrix
3. Cross validation for inner loop of conf_matrix()
4. Add column for data set name in csv

Shapley values for feature importance (which features are important) https://dalex.drwhy.ai/
(book https://ema.drwhy.ai/shapley.html)

========================================================================================================================

DATASETS:
adult:		https://archive.ics.uci.edu/ml/datasets/Adult
bc:			https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
btc heist:	https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset

========================================================================================================================

ISSUES:

- Every now and then get this error, no idea why, seems totally random.
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

"""
