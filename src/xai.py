# feature_importance
num_of_features = len(raw.columns)
min_train_x = process_data(raw, name, config, feature_name)
xai.feature_importance(min_train_x, process_data.grid, num_of_features)

plt.savefig('../results/feature_importance.png')
# def importance(dataset,min_train_x,model):
#     min_train_x = process_data()
#     num_of_features = len(dataset.columns)
#     xai.feature_importance(min_train_x, process_data.grid,num_of_features)


import pandas as pd
from sklearn import clone
import mainDataAI
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os

# CONFIG_FILE = '../config/run_config.json'
# configs_runs = get_configs()
# config = configs_runs[i]
#
# name = os.path.splitext(config['name'])[0]
def feature_importance(min_train_x,model,num_of_features):
    feat_imp = pd.DataFrame({'features': min_train_x.columns.tolist(), "mean_decrease_impurity": model.feature_importances_}).sort_values('mean_decrease_impurity', ascending=False)
    feat_imp = feat_imp.head(num_of_features)
    feat_imp.iplot(kind='bar',
                   y='mean_decrease_impurity',
                   x='features',
                   yTitle='Mean Decrease Impurity',
                   xTitle='Features',
                   title='Mean Decrease Impurity',
                  )

