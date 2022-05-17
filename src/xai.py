import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocess
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OrdinalEncoder
import pandas_profiling
import cufflinks as cf
from pdpbox import pdp, info_plots
import random
from sklearn.inspection import PartialDependenceDisplay
cf.set_config_file(sharing='public',theme='pearl',offline=False)
cf.go_offline()
from sklearn.inspection import plot_partial_dependence


configs_preprocessing = preprocess.get_configs()

for i in range(len(configs_preprocessing)):
    conf = configs_preprocessing[i]
    raw = pd.read_csv(conf["data_path"], sep=',', header=0)
    class_col = conf['class_column'] if conf['class_column'] != -1 else len(raw.columns) - 1
    raw.rename(columns={raw.columns[class_col]: 'Class'}, inplace=True)

    # replace NaN with median
    for col in raw.columns:
        if np.issubdtype(raw[col].dtype, np.number):
            raw[col].replace(np.NaN, raw[col].mean(), inplace=True)

    class_col = conf['class_column'] if conf['class_column'] != -1 else len(raw.columns) - 1
    raw.rename(columns={class_col: 'Class'}, inplace=True)

    X = raw.drop(columns=['Class'])
    y = raw['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    encoder = OrdinalEncoder()
    result = encoder.fit_transform(X)
    X_train_enc = encoder.transform(X_train)
    X_test_enc = encoder.transform(X_test)
    X_data = np.concatenate((X_train_enc,X_test_enc),0)

    forest = RandomForestClassifier(n_estimators=300)
    forest.fit(X_train_enc, y_train)
    y_pred = forest.predict(X_test_enc)
    feature_names = list(X.columns)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    plt.figure(figsize=(12, 12))
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    plt.savefig('../results/feature_importance/' + f'feature_importance_{conf["name"]}.png')

    result = permutation_importance(forest, X_test_enc, y_test, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=feature_names)

    fig, ax = plt.subplots()
    plt.figure(figsize=(12, 12))
    forest_importances.plot.bar(yerr=result.importances_std)
    plt.savefig('../results/feature_importance/' + f'permutation_importance_{conf["name"]}.png')

    def pdp_isolate(feature):
        pdp_age = pdp.pdp_isolate(
            model=forest, dataset=X_train_enc, model_features=X_train.columns, feature=feature
        )
        # PDP Plot
        fig, axes = pdp.pdp_plot(pdp_age, feature, plot_lines=False, center=False, frac_to_plot=0.5, plot_pts_dist=True,
                                 x_quantile=True, show_percentile=True)
        plt.savefig('../results/feature_importance/' + f'pdp_isolate_{conf["name"]}.png')


    pdp_isolate(X.columns[0])


    def pdp_interact(feature1, feature2):
        inter1 = pdp.pdp_interact(
            model=forest, dataset=X_train_enc, model_features=X_train.columns, features=[feature1, feature2]
        )
        fig, axes = pdp.pdp_interact_plot(
            pdp_interact_out=inter1, feature_names=[feature1, feature2], plot_type='contour', x_quantile=False,
            plot_pdp=False
        )
        axes['pdp_inter_ax'].set_yticklabels(feature1)
        axes['pdp_inter_ax'].set_xticklabels(feature2)
        plt.savefig('../results/feature_importance/' + f'pdp_interact_{conf["name"]}.png')


    pdp_interact(X.columns[0],X.columns[1])


    explainer = shap.TreeExplainer(forest, X_data)
    shap_values = explainer(X_data)
    shap.plots.waterfall(shap_values[0])
    plt.savefig('../results/feature_importance/' + f'shap_waterfall_{conf["name"]}.png')


    shap.summary_plot(shap_values, X_train_enc, feature_names=X.columns, plot_type="bar")
    plt.savefig('../results/feature_importance/' + f'shap_summary_{conf["name"]}.png')

    shap.dependence_plot(5, shap_values, X_train_enc, feature_names=X.columns)
    plt.savefig('../results/feature_importance/' + f'dependence_plot_{conf["name"]}.png')

    i = 18
    shap.force_plot(explainer.expected_value, shap_values[i], X_test[i], feature_names = X.columns)
    plt.savefig('../results/feature_importance/' + f'force_plot_{conf["name"]}.png')