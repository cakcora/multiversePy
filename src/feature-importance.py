import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocess
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
import cufflinks as cf
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import dalex as dx
from pdpbox import pdp
cf.set_config_file(sharing='public',theme='pearl',offline=False)
cf.go_offline()


configs_preprocessing = preprocess.get_configs()

for i in range(len(configs_preprocessing)):
    conf = configs_preprocessing[i]
    print(conf["name"])
    raw = pd.read_csv(conf["data_path"], sep=',', header=0)
    class_col = conf['class_column'] if conf['class_column'] != -1 else len(raw.columns) - 1
    raw.rename(columns={raw.columns[class_col]: 'Class'}, inplace=True)
    categorical_features = []
    cat_features = [col for col in list(raw.columns) if raw[col].dtypes == object]
    con_features = [col for col in list(raw.columns) if raw[col].dtypes == float]
    # replace NaN with median
    for col in raw.columns:
        if np.issubdtype(raw[col].dtype, np.number):
            raw[col].replace(np.NaN, raw[col].mean(), inplace=True)

    class_col = conf['class_column'] if conf['class_column'] != -1 else len(raw.columns) - 1
    raw.rename(columns={class_col: 'Class'}, inplace=True)

    missing = pd.DataFrame(round(raw.isna().sum() / raw.isna().count() * 100, 2))
    missing.rename(columns={0: 'Percentage of missing values'}, inplace=True)
    missing['Percentage of missing values'] = missing['Percentage of missing values'].map(str) + '%'
    raw = raw.fillna('nan')

    aux = raw.copy(deep=True)
    encoder = LabelEncoder()

    for col in aux[cat_features]:
        aux[col] = encoder.fit_transform(aux[col])
    num_features = [col for col in list(aux.columns) if aux[col].dtypes != object and col != 'Class']

    for col in aux[con_features]:
        aux[col] = encoder.fit_transform(aux[col])
    num_features = [col for col in list(aux.columns) if aux[col].dtypes != float and col != 'Class']

    scaler = MinMaxScaler()
    aux[num_features] = scaler.fit_transform(aux[num_features])

    X = aux.drop(columns=['Class'])
    y = aux['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    forest = RandomForestClassifier(n_estimators=300)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    feature_names = list(X.columns)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()

    forest_importances.plot.bar(yerr=std, ax=ax, figsize=(14, 12))

    ax.set_title("feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    plt.savefig('../results/feature_importance/' + f'feature_importance_{conf["name"]}.png')
    sortedSeries = forest_importances.sort_values(ascending=False)
    sortedSeries.drop(sortedSeries.index[5:], inplace=True)
    sortedSeries.to_csv(f'../results/feature_importance/feature_importance_{conf["name"]}.csv', mode='w')

    result = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    sortedPermutation = forest_importances.sort_values(ascending=False)
    sortedPermutation.drop(sortedPermutation.index[5:], inplace=True)
    sortedPermutation.to_csv(f'../results/feature_importance/permutation_importance_{conf["name"]}.csv', mode='w')
    fig, ax = plt.subplots()
    plt.figure(figsize=(14, 12))
    forest_importances.plot.bar(yerr=result.importances_std)
    plt.savefig('../results/feature_importance/' + f'permutation_importance_{conf["name"]}.png')

    # from sklearn.inspection import PartialDependenceDisplay
    # # def pdp_isolate(feature):
    # PartialDependenceDisplay.from_estimator(forest, X_train, [0, (0, 1)])
    # plt.savefig('../results/feature_importance/' + f'pdp_isolate_{conf["name"]}.png')

    # def pdp_interact(feature1, feature2):
    #     inter1 = pdp.pdp_interact(
    #         model=forest, dataset=X_train, model_features=X_train.columns, features=[feature1, feature2]
    #     )
    #     fig, axes = pdp.pdp_interact_plot(
    #         pdp_interact_out=inter1, feature_names=[feature1, feature2], plot_type='contour', x_quantile=False,
    #         plot_pdp=False
    #     )
    #     axes['pdp_inter_ax'].set_yticklabels(feature1)
    #     axes['pdp_inter_ax'].set_xticklabels(feature2)
    #     plt.savefig('../results/feature_importance/' + f'pdp_interact_{conf["name"]}.png')
    #
    # pdp_interact(X.columns[0],X.columns[1])



