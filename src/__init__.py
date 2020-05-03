import random

# from sklearn.tree import export_graphviz
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

sampleSize = 1000


def mineTrees(rf_model):
    result = pd.DataFrame(index=np.arange(0, rf_model.n_estimators),
                          columns=['nodes', 'edges', 'diameter', 'weak_components', 'strong_components',
                                   'node_connectivity', 'mean_hub_score', 'mean_auth_score',
                                   'median_degree', 'mean_degree'])
    for t in range(0, rf_model.n_estimators):
        # print("Tree " + str(t) + " is processing")
        tree = rf_model.estimators_[t]
        graph = nx.DiGraph()

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
                    graph.add_edge(node, features[r_child])

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
    target_data = target_data
    if percentage > 0:
        l = len(target_data.index)
        end = int(percentage * l / 100)
        if message:
            print("out of " + str(l) + " rows, labels of " + str(end) + " will be flipped")
        unique_vals = target_data.Class.unique()
        if len(unique_vals) <= 1:
            print("ERROR: Dataset contains a single label for the Class feature")
        else:
            for i in range(1, end):
                current_label = target_data.Class[i]
                new_label = current_label
                while new_label == current_label:
                    new_label = str(random.choice(unique_vals))
                target_data.Class[i] = new_label  # gives SettingWithCopyWarning
    return target_data


census_file = "C:/Users/akkar/Dropbox/Academic/Manitoba/Data Science of ML Models/Datasets/Census/census-income.data"
data = pd.read_csv(census_file, header=None).sample(n=sampleSize)
data = data.rename(columns={data.columns[41]: "Class"})
factor = pd.factorize(data['Class'])
data.Class = factor[0]
definitions = factor[1]
# print(data.Class.head())
# print(definitions)
print(data.head(1))
# list(data.columns)
# data.describe()

major_max = 1
minor_max = 5
n_est = 500
print(str(len(data)) + " data rows.")
for major in range(0, major_max):
    data = poison(data, major)
    df = pd.DataFrame()
    for minor in range(0, minor_max):
        data2 = poison(data, major, True)
        print("\tmajor:", major, " minor:", minor)
        # one-hot encoding of the data (except for the Class variable)
        dataTrain = pd.get_dummies(data2.loc[:, data2.columns != 'Class'])

        rf = RandomForestRegressor(n_estimators=n_est, random_state=42)
        rf.fit(dataTrain, data2.Class)
        result = mineTrees(rf)
        result['minor'] = minor
        df = df.append(result)
    rf2 = RandomForestRegressor(n_estimators=n_est)
    X = df.loc[:, df.columns != 'minor']
    y = df.minor
    rf2.fit(X, y)

    importances = rf2.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf2.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importance of the forest
    plt.figure()
    plt.title("Feature importance")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
