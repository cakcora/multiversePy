import networkx as nx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import json
import warnings
import os
import glob
from time import perf_counter
import preprocess
# from sklearn.tree import export_graphviz  # used in comment
import warnings
warnings.filterwarnings("ignore")

CONFIG_FILE = '../config/run_config.json'


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
	poison a percentage of a series (target_data) and return a new dataframe
	:param target_data:     series to poison
	:param percentage:      percentage of dataframe to poison
	:param message:         print a status message
	:return:                poisoned series
	"""
	poisoned_data = target_data.copy()  # avoid poisoning original data

	if percentage > 0:
		length = len(poisoned_data.index)
		num_to_poison = int(percentage * length / 100)

		if message:
			print(f'out of {length} rows, labels of {num_to_poison} will be flipped')
		unique_vals = poisoned_data.unique()

		if len(unique_vals) <= 1:
			print("ERROR: Dataset contains a single label for the Class feature")
		else:
			# don't poison from the beginning. pick i randomly and make sure you don't take same i two times
			for i in np.random.choice(poisoned_data.index.array, num_to_poison, replace=False):  # random indices
				new_label = poisoned_data.loc[i]
				while new_label == poisoned_data.loc[i]:
					new_label = np.random.choice(unique_vals)
				poisoned_data.at[i] = new_label  # fixes SettingWithCopyWarning

	return poisoned_data


def conf_matrix(prepped_data, name, config):
	"""
	:param prepped_data:        encoded dataframe to train and test forests
	:param name:                name of dataset
	:param config:              configuration dictionary
	"""
	entropy_list = []  # Will hold tuples of the form (Major, Entropy Value)
	accuracy_list = []  # tuples: (major, minor, validation accuracy, test accuracy)

	# first test-train split
	y = prepped_data['Class']
	try:
		min_train_x, min_x_test, min_train_y, min_y_test = train_test_split(
			prepped_data.drop(columns=['Class']),
			y,
			random_state=1,
			stratify=y
		)
	except ValueError as e:
		warnings.warn(str(e), UserWarning)
		min_train_x, min_x_test, min_train_y, min_y_test = train_test_split(  # workaround for BTC Heist
			prepped_data.drop(columns=['Class']),
			y,
			random_state=1,
			stratify=None  # do not stratify
		)

	major_max = min(config['major_max'], max(100 // len(pd.unique(prepped_data['Class'])), 1))  # cap major max
	for major in range(0, major_max+1, 5):
		maj_poison_y = poison(min_train_y, major)
		df = pd.DataFrame()

		for minor in range(0, config['minor_max']+1,5):
			print("\nmajor:", major, " minor:", minor)
			min_poison_y = poison(maj_poison_y, minor, True)

			# config['param_grid'] = {}  # speed up for debugging by not optimizing
			grid = GridSearchCV(
				estimator=RandomForestClassifier(n_estimators=config['n_estimators']),
				param_grid=config['param_grid'],
				cv=config['n_cv_folds'],
				n_jobs=10
			)
			grid.fit(min_train_x, min_poison_y)

			minor_test_accuracy = accuracy_score(min_y_test, grid.best_estimator_.predict(min_x_test))
			accuracy_list.append((major, minor, grid.best_score_, minor_test_accuracy))

			print(f'\tOptimized parameters = {grid.best_params_}')  # TODO: log this
			print(f'\tValidation accuracy:{grid.best_score_:10.5f}')
			print(f'\tTest accuracy:{minor_test_accuracy:16.5f}')

			result = mineTrees(grid.best_estimator_)
			result['minor'] = minor
			df = df.append(result)

		y = df['minor']
		x = df.drop('minor', axis=1)

		# Second test-train split
		x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, stratify=y)

		rf_second_level = RandomForestClassifier(n_estimators=config['n_estimators'])
		rf_second_level.fit(x_train, y_train)

		y_pred_test = rf_second_level.predict(x_test)  # test second level model

		# The commented line is what was added during the meeting, however it was
		# throwing errors. This seems to be corrected now?
		# matrix = confusion_matrix(test_data.minor, y_pred_test)
		matrix = confusion_matrix(y_test, y_pred_test)

		entropy = scipy.stats.entropy(matrix.flatten())
		entropy_list.append((major, entropy))

		if major == 0:
			first_matrix = matrix

	entropy_df = pd.DataFrame(entropy_list, columns=['major', f'{name}_entropy']).set_index('major')
	entropy_df.to_csv(f'../results/entropy/{name}.csv')

	if not os.path.exists(config['temp_csv_dir']):
		os.makedirs(config['temp_csv_dir'], exist_ok=True)
	entropy_df.to_csv(f'{config["temp_csv_dir"]}{name}.csv')


def plot2(df_entropy1, df_entropy2):
	"""
	Plots the recorded entropy values and the final confusion matrix.
	:param df_entropy:      Entropy dataframe. Should contain index col of Major values, 'Entropy' column of entropy values.
	:param matrix:          Last confusion matrix
	:param first_matrix     First confusion matrix
	:param name:            name of dataset
	:param config           Configuration dictionary
	"""
	sns.set(rc={'figure.figsize': (20, 10)})

	ax = sns.regplot(x=df_entropy1.index, y=df_entropy1['carevaluation'])
	ax = sns.regplot(x=df_entropy2.index, y=df_entropy2['auctionverification'])

	ax.set(title="Conf Matrix Entropy", xlabel="Major", ylabel="Entropy")

	plt.savefig('../results/entropy/4ent.png')
	plt.close()


def get_configs():
	"""
	:return:    list of configuration dictionaries for datasets
	"""
	with open(CONFIG_FILE, 'rt') as f:
		config = json.load(f)

	datasets = config.pop('datasets')
	return [{'dataset': dataset, **config} for dataset in datasets]  # individual configs for datasets


def get_data(config):
	"""
	:param config: dataset configuration
	:return:        dataframe
	"""
	data = pd.read_csv(config['preprocessed_dir'] + config['dataset'], header=None)
	data.rename(columns={len(data.columns) - 1: 'Class'}, inplace=True)  # rename last col to 'Class'


	if 0 < config['sample_size'] <= len(data.index):
		return data.sample(n=config['sample_size'])
	return data


def run_dataset(config):
	"""
	load dataset and generate confusion matrix results
	:param config:      dataset configuration dictionary
	"""
	name = os.path.splitext(config['dataset'])[0]
	print("PROCESSING DATA SET: " + name)
	data = get_data(config)
	conf_matrix(data, name, config)


def main():
	# preprocess.preprocess(dataset_names=None)  # None means all datasets
	os.chdir(os.path.dirname(os.path.realpath(__file__)))

	configs = get_configs()
	for config in configs:
		run_dataset(config)

	combine_entropy_data(configs[0])
	entropy_plot(configs[0])


if __name__ == '__main__':
	t = perf_counter()
	main()
	print(f'Time: {perf_counter() - t:.2f}')
