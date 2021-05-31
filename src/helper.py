import random
# from sklearn.tree import export_graphviz
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import math
import scipy.stats


def mineTrees(rf_model):
    result = pd.DataFrame(index=np.arange(0, rf_model.n_estimators),
                          columns=['nodes', 'edges', 'diameter', 'weak_components', 'strong_components',
                                   'node_connectivity', 'mean_hub_score', 'mean_auth_score',
                                   'median_degree', 'mean_degree'])
    for t in range(0, rf_model.n_estimators):
        # print("Tree " + str(t) + " is processing")
        tree = rf_model.estimators_[t]
        graph = nx.DiGraph() # Multiple edges are not allowed

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
                    graph.add_edge(node, features[r_child]) #compare the graph with the original decision tree to make that the graph is correct

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
    if percentage > 0:
        l = len(target_data.index)
        end = int(percentage * l / 100)
        if message:
            print("out of " + str(l) + " rows, labels of " + str(end) + " will be flipped")
        unique_vals = target_data.Class.unique()
        if len(unique_vals) <= 1:
            print("ERROR: Dataset contains a single label for the Class feature")
        else:
            #don't poison from the beginning. pick i randomly and make sure you don't take same i two times
            for i in range(1, end):
                current_label = target_data.Class[i]
                new_label = current_label
                while new_label == current_label:
                    new_label = str(random.choice(unique_vals))
                target_data.Class[i] = new_label  # gives SettingWithCopyWarning
    return target_data

def conf_matrix(data, major_max, minor_max):
    n_est = 100
    le = preprocessing.LabelEncoder()
    balance_data = data.apply(le.fit_transform)

    for major in range(0, major_max):
        dataPoisoned = poison(balance_data, major)
        df = pd.DataFrame()
        for minor in range(0, minor_max):
            dataPoisoned2 = poison(dataPoisoned, major, True)
            print("\tmajor:", major, " minor:", minor)
            # one-hot encoding of the data (except for the Class variable)
            dataTrain = pd.get_dummies(dataPoisoned2.loc[:, dataPoisoned2.columns != 'Class'])
            rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
            rf_fit = rf.fit(dataTrain, dataPoisoned2.Class)
            result = mineTrees(rf_fit)
            result['minor'] = minor
            df = df.append(result)

        rf2 = RandomForestClassifier(n_estimators=n_est)
        y = dataPoisoned2['Class']

        X = dataPoisoned2.drop('Class', axis=1)
        print(y.value_counts())
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
        forest = RandomForestClassifier()
        rf2_fit = rf2.fit(X_train, y_train)
        y_pred_test = rf2_fit.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred_test)
        print(matrix)
        entropy = scipy.stats.entropy(matrix)
        print(entropy)
        #entropies = pd.DataFrame()


    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2)
    plt.show()