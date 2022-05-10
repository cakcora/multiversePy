import pandas as pd
from sklearn import clone
import mainDataAI
import numpy as np
from sklearn.model_selection import cross_val_score


num_of_features = len(mainDataAI.raw.columns)
def feature_importance(model,num_of_features):
    feat_imp = pd.DataFrame({'features': mainDataAI.min_train_x.columns.tolist(), "mean_decrease_impurity": mainDataAI.grid.feature_importances_}).sort_values('mean_decrease_impurity', ascending=False)
    feat_imp = feat_imp.head(num_of_features)
    feat_imp.iplot(kind='bar',
                   y='mean_decrease_impurity',
                   x='features',
                   yTitle='Mean Decrease Impurity',
                   xTitle='Features',
                   title='Mean Decrease Impurity',
                  )

# get the feature importances from each tree and then visualize the
# distributions as boxplots
#     all_feat_imp_df = pd.DataFrame(data=[tree.feature_importances_ for tree in model],
#                                columns=mainDataAI.min_train_x.columns)
#     order_column = all_feat_imp_df.mean(axis=0).sort_values(ascending=False).index.tolist()
#
#     all_feat_imp_df[order_column[:-1]].iplot(kind='box', xTitle='Features', yTitle='Mean Decease Impurity')


# def dropcol_importances(model, X_train, y_train, cv=3):
#    model_ = clone(model)
#     model_.random_state = 42
#     baseline = cross_val_score(model_, X_train, y_train, scoring='accuracy', cv=cv)
#     imp = []
#     for col in X_train.columns:
#         X = X_train.drop(col, axis=1)
#         model_ = clone(model)
#         model_.random_state = 42
#         oob = cross_val_score(rf_, X, y_train, scoring='accuracy', cv=cv)
#         imp.append(baseline - oob)
#     imp = np.array(imp)
#
#     importance = pd.DataFrame(
#         imp, index=X_train.columns)
#     importance.columns = ["cv_{}".format(i) for i in range(cv)]
#     return importance
#

# drop_col_imp = dropcol_importances(rf, X_train, y_train, cv=50)
# drop_col_importance = pd.DataFrame(
#     {'features': X_train.columns.tolist(), "drop_col_importance": drop_col_imp.mean(axis=1).values}).sort_values(
#     'drop_col_importance', ascending=False)
# drop_col_importance = drop_col_importance.head(25)
# drop_col_importance.iplot(kind='bar',
#                           y='drop_col_importance',
#                           x='features',
#                           yTitle='Drop Column Importance',
#                           xTitle='Features',
#                           title='Drop Column Importances',
#                           )
#
# all_feat_imp_df = drop_col_imp.T
# order_column = all_feat_imp_df.mean(axis=0).sort_values(ascending=False).index.tolist()
#
# all_feat_imp_df[order_column[:-1]].iplot(kind='box', xTitle='Features', yTitle='Drop Column Importance')
#
#
# from mlxtend.evaluate import feature_importance_permutation
# #This takes sometime. You can reduce this number to make the process faster
# num_rounds = 50
# imp_vals, all_trials = feature_importance_permutation(
#     predict_method=rf.predict,
#     X=X_test.values,
#     y=y_test.values,
#     metric='accuracy',
#     num_rounds=num_rounds,
#     seed=1)
# permutation_importance = pd.DataFrame({'features': X_train.columns.tolist(), "permutation_importance": imp_vals}).sort_values('permutation_importance', ascending=False)
# permutation_importance = permutation_importance.head(25)
# permutation_importance.iplot(kind='bar',
#                y='permutation_importance',
#                x='features',
#                yTitle='Permutation Importance',
#                xTitle='Features',
#                title='Permutation Importances',
#               )
#
# all_feat_imp_df = pd.DataFrame(data=np.transpose(all_trials),
#                                columns=X_train.columns, index=range(0, num_rounds))
# order_column = all_feat_imp_df.mean(axis=0).sort_values(ascending=False).index.tolist()
#
# all_feat_imp_df[order_column[:25]].iplot(kind='box', xTitle='Features', yTitle='Permutation Importance')
#
#
# from pdpbox import pdp, info_plots
# pdp_age = pdp.pdp_isolate(
#     model=rf, dataset=X_train, model_features=X_train.columns, feature='age'
# )
# #PDP Plot
# fig, axes = pdp.pdp_plot(pdp_age, 'Age', plot_lines=False, center=False, frac_to_plot=0.5, plot_pts_dist=True,x_quantile=True, show_percentile=True)
# #ICE Plot
# fig, axes = pdp.pdp_plot(pdp_age, 'Age', plot_lines=True, center=False, frac_to_plot=0.5, plot_pts_dist=True,x_quantile=True, show_percentile=True)
#
#
# # All the one-hot variables for the occupation feature
# occupation_features = ['occupation_ ?', 'occupation_ Adm-clerical', 'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving']
# #Notice we are passing the list of features as a list with the feature parameter
# pdp_occupation = pdp.pdp_isolate(
#     model=rf, dataset=X_train, model_features=X_train.columns,
#     feature=occupation_features
# )
# #PDP
# fig, axes = pdp.pdp_plot(pdp_occupation, 'Occupation', center = False, plot_pts_dist=True)
# #Processing the plot for aesthetics
# _ = axes['pdp_ax']['_pdp_ax'].set_xticklabels([col.replace("occupation_","") for col in occupation_features])
# axes['pdp_ax']['_pdp_ax'].tick_params(axis='x', rotation=45)
# bounds = axes['pdp_ax']['_count_ax'].get_position().bounds
# axes['pdp_ax']['_count_ax'].set_position([bounds[0], 0, bounds[2], bounds[3]])
# _ = axes['pdp_ax']['_count_ax'].set_xticklabels([])
#
#
# # Age and Education
# inter1 = pdp.pdp_interact(
#     model=rf, dataset=X_train, model_features=X_train.columns, features=['age', 'education_num']
# )
# fig, axes = pdp.pdp_interact_plot(
#     pdp_interact_out=inter1, feature_names=['age', 'education_num'], plot_type='contour', x_quantile=False, plot_pdp=False
# )
# axes['pdp_inter_ax'].set_yticklabels([edu_map.get(col) for col in axes['pdp_inter_ax'].get_yticks()])
#
# # PDP Sex
# pdp_sex = pdp.pdp_isolate(
#     model=rf, dataset=X_train, model_features=X_train.columns, feature='sex'
# )
# fig, axes = pdp.pdp_plot(pdp_sex, 'Sex', center=False, plot_pts_dist=True)
# _ = axes['pdp_ax']['_pdp_ax'].set_xticklabels(sex_le.inverse_transform(axes['pdp_ax']['_pdp_ax'].get_xticks()))
#
# # marital_status and sex
# inter1 = pdp.pdp_interact(
#     model=rf, dataset=X_train, model_features=X_train.columns, features=['marital_status', 'sex']
# )
# fig, axes = pdp.pdp_interact_plot(
#     pdp_interact_out=inter1, feature_names=['marital_status', 'sex'], plot_type='grid', x_quantile=False, plot_pdp=False
# )
# axes['pdp_inter_ax'].set_xticklabels(marital_le.inverse_transform(axes['pdp_inter_ax'].get_xticks()))
# axes['pdp_inter_ax'].set_yticklabels(sex_le.inverse_transform(axes['pdp_inter_ax'].get_yticks()))
# xai.feature_importance(process_data.grid)