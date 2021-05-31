import random
from helper import *
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


major_max = 2
minor_max = 2
class_column = 14
sampleSize = 50

class_names = "C:/Users/poupa/Downloads/multiversePy-last/data/census/adult.names"
census_file = "C:/Users/poupa/Downloads/multiversePy-last/data/census/adult.data"

data = pd.read_csv(census_file, header=None).sample(n=sampleSize)
data = data.rename(columns={data.columns[class_column]: "Class"})

factor = pd.factorize(data['Class'])
data.Class = factor[0]

conf_matrix(data, major_max, minor_max)

