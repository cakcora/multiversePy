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
        mean_hub_score = np.mean(list(hubs.values()))
        mean_auth_score = np.mean(list(authorities.values()))
        nodes = nx.number_of_nodes(graph)
        diameter = nx.diameter(nx.to_undirected(graph))
        edges = nx.number_of_edges(graph)
        strong_comp = nx.number_strongly_connected_components(graph)
        weak_comp = nx.number_weakly_connected_components(graph)
        degrees = nx.average_degree_connectivity(graph, target="in")
        avg_in_degree = np.mean(list(degrees))
        median_in_degree = np.median(list(degrees))
        node_connectivity = nx.average_node_connectivity(graph)
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
            print(f'out of {length} rows, labels of {num_to_poison} will be flipped')  #
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


def conf_matrix(data, major_max, minor_max, n_est=100):
    """
    :param data:            dataframe to train forests on (unprocessed)
    :param major_max:       major axis iterations
    :param minor_max:       minor axis iterations
    :param n_est:           n_estimators to use for random forest classifiers
    """

    # encode (all) categorical data into numeric categories (not one-hot)
    le = preprocessing.LabelEncoder()
    balance_data = data.apply(le.fit_transform)

    for major in range(0, major_max):
        dataPoisoned = poison(balance_data, major)
        df = pd.DataFrame()  # this is modified but never accessed

        for minor in range(0, minor_max):
            dataPoisoned2 = poison(dataPoisoned, major, True)
            print("\tmajor:", major, " minor:", minor)

            # one-hot encoding of the data (except for the Class variable)
            dataTrain = pd.get_dummies(dataPoisoned2.loc[:, dataPoisoned2.columns != 'Class'])

            # TODO: fix one-hot encoding
            assert dataTrain.equals(dataPoisoned2.loc[:, dataPoisoned2.columns != 'Class'])  # get_dummies did nothing

            rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
            rf.fit(dataTrain, dataPoisoned2.Class)

            result = mineTrees(rf)
            result['minor'] = minor
            df.append(result)

        rf2 = RandomForestClassifier(n_estimators=n_est)
        y = dataPoisoned2['Class']

        X = dataPoisoned2.drop('Class', axis=1)
        print(y.value_counts())
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

        rf2.fit(X_train, y_train)
        y_pred_test = rf2.predict(X_test)

        matrix = confusion_matrix(y_test, y_pred_test)
        print(matrix)

        entropy = scipy.stats.entropy(matrix)
        print(entropy)
        # entropies = pd.DataFrame()

    sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
                cmap=plt.cm.Greens, linewidths=0.2)  # show last matrix
    plt.show()


if __name__ == '__main__':
    # load configs
    config_file = "../config.json"  # relative path to config file
    with open(config_file, 'rt') as f:
        config = json.load(f)

        class_names = config.get('class_names')  # unused
        census_file = config.get('census_file')
        class_column = config.get('class_column')

        major_max = config.get('major_max')
        minor_max = config.get('minor_max')
        sample_size = config.get('sample_size')
        n_estimators = config.get('n_estimators')

    # make the data
    data = pd.read_csv(census_file, header=None).sample(n=sample_size)  # save clean data for accuracy tests here?
    data = data.rename(columns={data.columns[class_column]: "Class"})

    conf_matrix(data, major_max, minor_max, n_est=n_estimators)
