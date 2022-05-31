import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from lime.lime_tabular import LimeTabularExplainer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import time
import shap
from lime.lime_tabular import LimeTabularExplainer
import seaborn as sns

# df = pd.read_csv("../data/breast-cancer-wisconsin.data", header=None)
# df.columns = ['id', 'Clump_thickness', 'Uniformaty_cell_size', 'Uniformaty_cell_shape', 'Marginal_adhesion', 'Single_Epithelial_Cell_Size',
#                  'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli ', 'Mitoses', 'Class']

df = pd.read_csv("../data/Adult/adult.csv", header=None)
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
              'race ', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']
cat_features = [col for col in list(df.columns) if df[col].dtypes == object]

encoder = LabelEncoder()
for col in df[cat_features]:
    df[col] = encoder.fit_transform(df[col])

features, labels = df.drop('class', axis=1), df['class'] #observes and label
print(df.shape)

#set parameters for the plot
def setup_plot():
  plt.rcParams["axes.grid.axis"] ="y"
  plt.rcParams["axes.grid"] = True
  plt.rc('grid', linestyle="dashed", color='lightgrey', linewidth=1)
  plt.rcParams["xtick.labelsize"] = 12
  plt.rcParams["ytick.labelsize"]  = 10

## train a model
def train_model(model, data, labels):
  x_train, x_test, y_train, y_test = train_test_split(data, labels.values, random_state=2)

  pipe = Pipeline([('scaler', StandardScaler()),('clf', model["clf"])])
  start_time = time.time()
  pipe.fit(x_train, y_train)
  train_time = time.time() - start_time

  train_accuracy =  pipe.score(x_train, y_train)
  test_accuracy = pipe.score(x_test, y_test)
  details = {"name": model["name"], "train_accuracy":train_accuracy, "test_accuracy":test_accuracy, "train_time": train_time, "model": pipe}
  return details

## train four models and compare them
trained_models = [] #  keep track of all details for models we train
models = [
      {"name": "logistic regression", "clf": LogisticRegressionCV()},
      {"name": "random forest", "clf": RandomForestClassifier(n_estimators=200)},
      {"name": "Gradient Boosting", "clf": GradientBoostingClassifier(n_estimators=200)},
      {"name": "Naive Bayes", "clf": GaussianNB()}
      ]

for model in models:
  model_details = train_model(model, features, labels)
  trained_models.append(model_details)

# visualize accuracy and run time
setup_plot()
model_df = pd.DataFrame(trained_models)
#print(trained_models[1])
model_df.sort_values("test_accuracy", inplace=True)
ax = model_df[["train_accuracy","test_accuracy", "name"]].plot(kind="line", x="name", figsize=(10,5), title="Classifier Performance Sorted by Test Accuracy")
ax.legend(["Train Accuracy", "Test Accuracy"])

ax.title.set_size(20)
plt.box(False)
plt.savefig('../results/xai/Accuracy')


model_df.sort_values("train_time", inplace=True)
ax= model_df[["train_time","name"]].plot(kind="line", x="name", figsize=(10,5), grid=True, title="Classifier Training Time (seconds)")
ax.title.set_size(20)
ax.legend(["Train Time"])
plt.box(False)
plt.savefig('../results/xai/Train_time')


current_data = features
print(current_data.columns)
plt.figure(figsize=(15,6))
X_train, X_test, y_train, y_test = train_test_split(current_data, labels.values, random_state=2)
# logistic_reg_coeff = trained_models[0]["model"]["clf"].coef_
color_list =  sns.color_palette("dark", len(current_data.columns))
# top_x = 9
# logistic_reg_coeff = trained_models[0]["model"]["clf"].coef_[0]
# idx = np.argsort(np.abs(logistic_reg_coeff))[::-1]
# lreg_ax = plt.barh(current_data.columns[idx[:top_x]][::-1], logistic_reg_coeff[idx[:top_x]][::-1])
# for i,bar in enumerate(lreg_ax):
#   bar.set_color(color_list[idx[:top_x][::-1][i]])
#   plt.box(False)
#   lr_title = plt.suptitle("Logistic Regression. Top " + str(top_x) + " Coefficients.", fontsize=20, fontweight="normal")


X_train, X_test, y_train, y_test = train_test_split(current_data, labels, random_state=2)

def get_lime_explainer(model, data, labels):
    cat_feat_ix = [i for i, c in enumerate(data.columns) if pd.api.types.is_categorical_dtype(data[c])]
    feat_names = list(data.columns)
    class_names = list(labels.unique())
    scaler = model["model"]["scaler"]
    data = scaler.transform(data)  # scale data to reflect train time scaling
    lime_explainer = LimeTabularExplainer(data,
                                          feature_names=feat_names,
                                          class_names=class_names,
                                          categorical_features=cat_feat_ix,
                                          mode="classification",
                                          discretize_continuous=False
                                          )
    return lime_explainer


def lime_explain(explainer, data, predict_method, num_features):
    explanation = explainer.explain_instance(data, predict_method, num_features=num_features)
    return explanation

lime_data_explainations = []
lime_metrics = []
lime_explanation_time = []
feat_names = list(current_data.columns)
test_data_index = 0
for current_model in trained_models:
    scaler = current_model["model"]["scaler"]
    scaled_test_data = scaler.transform(X_test)
    predict_method = current_model["model"]["clf"].predict_proba

    start_time = time.time()
    # explain first sample from test data
    lime_explainer = get_lime_explainer(current_model, X_train, y_train)
    explanation = lime_explain(lime_explainer, scaled_test_data[test_data_index], predict_method, 9)
    # print("The " + str(test_data_index) + "th instance in the testset with true label: " + labels[
    #     test_data_index] + "; And model: " + current_model["name"])
    explanation.show_in_notebook(show_table=True)
    elapsed_time = time.time() - start_time

    ex_holder = {}
    for feat_index, ex in explanation.as_map()[1]:
        ex_holder[feat_names[feat_index]] = ex

    lime_data_explainations.append(ex_holder)
    actual_pred = predict_method(scaled_test_data[test_data_index].reshape(1, -1))
    perc_pred_diff = abs(actual_pred[0][1] - explanation.local_pred[0])
    lime_explanation_time.append({"time": elapsed_time, "model": current_model["name"]})
    lime_metrics.append({"lime class1": explanation.local_pred[0], "actual class1": actual_pred[0][1],
                         "class_diff": round(perc_pred_diff, 3), "model": current_model["name"]})

print(lime_metrics)

# def plot_lime_exp(fig, fig_index, exp_data, title):
#   features =  list(exp_data.keys())[::-1]
#   explanations = list(exp_data.values())[::-1]
#   ax = fig.add_subplot(fig_index)
#   lime_bar = ax.barh( features, explanations )
#   ax.set_title(title, fontsize = 20)
#   for i,bar in enumerate(lime_bar):
#     bar.set_color(color_list[list(current_data.columns).index(features[i])])
#     plt.box(False)
#     plt.savefig('../results/xai/plot_lime')


fig = plt.figure(figsize=(19,8))

# Plot lime explanations for trained models
for i, dex in enumerate(lime_data_explainations):
  fig_index = int("23" + str(i+1))
  plot_lime_exp(fig, fig_index, lime_data_explainations[i], trained_models[i]["name"])

plt.suptitle( " LIME Explanation for single test data instance", fontsize=20, fontweight="normal")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('../results/xai/LIME_for_single_data')

# Plot run time for explanations
lx_df = pd.DataFrame(lime_explanation_time)
lx_df.sort_values("time", inplace=True)
setup_plot()
lx_ax = lx_df.plot(kind="line", x="model", title="Runtime (seconds) for single test data instance LIME explanation", figsize=(22,6))
lx_ax.title.set_size(20)
lx_ax.legend(["Run time"])
plt.box(False)
plt.savefig('../results/xai/Lime_run_time')

# Plot run time for explanations
print(lime_metrics)

lime_metrics_df = pd.DataFrame(lime_metrics)
lime_metrics_df_ax = lime_metrics_df[["lime class1", "actual class1", "model"]].plot(kind="line", x="model", title="LIME Actual Prediction vs Local Prediction ", figsize=(22,6))
lime_metrics_df_ax.title.set_size(20)
lime_metrics_df_ax.legend(["Lime Local Prediction", "Actual Prediction"])
plt.box(False)
plt.savefig('../results/xai/Lime_Actual_Local')

current_model = trained_models[1] #the second model is RF
clf = current_model["model"]["clf"]
scaler = current_model["model"]["scaler"]

scaled_train_data = scaler.transform(X_train)
scaled_test_data = scaler.transform(X_test)
subsampled_test_data =scaled_test_data[test_data_index].reshape(1,-1)

# explain first sample from test data
start_time = time.time()
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(subsampled_test_data)
elapsed_time = time.time() - start_time

print("Tree Explainer SHAP run time", round(elapsed_time,3) , " seconds. ", current_model["name"])
print("SHAP expected value", explainer.expected_value)
print("Model mean value", clf.predict_proba(scaled_train_data).mean(axis=0))
print("Model prediction for test data", clf.predict_proba(subsampled_test_data))
shap.initjs()
pred_ind = 0
shap.force_plot(explainer.expected_value[1], shap_values[1][0], subsampled_test_data[0], feature_names=X_train.columns)
plt.savefig('../results/xai/Shap_force_plot')

shap.initjs()
plt.rc('font', size=15)
shap.summary_plot(shap_values, subsampled_test_data, show=False, plot_type="bar", plot_size=(15, 10), feature_names=X_train.columns, max_display=10)
plt.title('Shap')
plt.savefig('../results/xai/Shap_summary_plot')