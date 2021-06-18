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

        node_connectivity = nx.average_node_connectivity(graph)  # how well the graph is connected

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


def conf_matrix(data, test, major_max, minor_max, n_est=100):
    """
    :param data:            dataframe to train forests
    :param test:            dataframe to test forests
    :param major_max:       number of major axis iterations
    :param minor_max:       number of minor axis iterations
    :param n_est:           n_estimators to use for random forest classifiers
    """

    for major in range(0, major_max):
        dataPoisoned = poison(data, major)
        df = pd.DataFrame()  # this is modified but never accessed

        for minor in range(0, minor_max):
            dataPoisoned2 = poison(dataPoisoned, major, True)
            print("\tmajor:", major, " minor:", minor)

            rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
            rf.fit(dataPoisoned2.drop('Class', axis=1), dataPoisoned2['Class'])  # train w/ poisoned2

            result = mineTrees(rf)
            result['minor'] = minor
            df.append(result)

        y = dataPoisoned2['Class']
        X = dataPoisoned2.drop('Class', axis=1)
        print(y.value_counts())

        rf2 = RandomForestClassifier(n_estimators=n_est)
        rf2.fit(X, y)

        y_pred_test = rf2.predict(test.drop('Class', axis=1))  # test model against un-poisoned data

        matrix = confusion_matrix(test.Class, y_pred_test)
        print(matrix)

        entropy = scipy.stats.entropy(matrix)
        print(entropy)
        # entropies = pd.DataFrame()

    sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
                cmap=plt.cm.Greens, linewidths=0.2)  # show last matrix
    plt.show()


def read_sample_split_data(conf):
    """
    read and preprocess data according to settings in conf
    :param conf:        dictionary of configurations
    :return:            tuple of dataframes: (training-data, testing-data)
    """
    raw = pd.read_csv(conf['census_file'], names=conf['column_names'], header=None)
    raw = raw.sample(n=conf['sample_size'])

    # one-hot encoding of the raw (except for the Class variable)
    data = pd.get_dummies(raw.loc[:, raw.columns != 'Class'])

    le = preprocessing.LabelEncoder()  # encode Class variable numerically
    data['Class'] = le.fit_transform(raw['Class'])

    test = data.sample(frac=config['test_fraction'])
    data.drop(index=data.index.intersection(test.index), inplace=True)  # remove rows that are in test

    return data, test


if __name__ == '__main__':
    config_file = "../config.json"  # relative path to config file
    with open(config_file, 'rt') as f:
        config = json.load(f)

    data, test = read_sample_split_data(config)
    conf_matrix(data, test, config['major_max'], config['minor_max'], n_est=config['n_estimators'])
