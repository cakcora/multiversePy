import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df_entropy1 = pd.read_csv('../results/entropy/eyestate.csv', header=0)
df_entropy2 = pd.read_csv('../results/entropy/firmteacherclavedirection.csv', header=0)
df_entropy3 = pd.read_csv('../results/entropy/gendergap.csv', header=0)
df_entropy4 = pd.read_csv('../results/entropy/gesturephase.csv', header=0)
df = pd.DataFrame({"labels": ['Q1', 'Q2', 'Q3', 'Q4']})

def plot(df_ent1, df_ent2, df_ent3, df_ent4):
	"""
	Plots the recorded entropy values and the final confusion matrix.
	:param df_entropy:      Entropy dataframe. Should contain index col of Major values, 'Entropy' column of entropy values.
	:param matrix:          Last confusion matrix
	:param first_matrix     First confusion matrix
	:param name:            name of dataset
	:param config           Configuration dictionary
	"""
	# sns.set(rc={'figure.figsize': (20, 10)})
	# ax = sns.regplot(data=df)
	# ax = sns.regplot(x=df_ent1.index, y=df_ent1.index,fit_reg=False, ci=None,label='EGG Eye State')
	# ax = sns.regplot(x=df_ent2.index, y=df_ent2['firmteacherclavedirection_entropy'],fit_reg=False,ci=None, label='Firm Tacher Clave Direction')
	# ax = sns.regplot(x=df_ent3.index, y=df_ent3['gendergap_entropy'],fit_reg=False,ci=None, label='Gender Gap')
	# ax = sns.regplot(x=df_ent4.index, y=df_ent4['gesturephase_entropy'],fit_reg=False,ci=None, label='Gesture Phase')
	# ax.legend()
	# ax.set(title="Conf Matrix Entropy", xlabel="Major", ylabel="Entropy")

	fig = px.line(df_entropy4, x=df_ent4.index, y=df_ent4['gesturephase_entropy'], color="country", text="year")
	fig.update_traces(textposition="bottom right")
	fig.show()

	plt.savefig('../results/entropy/entropy2.png')
	plt.close()

plot(df_entropy1, df_entropy2, df_entropy3, df_entropy4)
