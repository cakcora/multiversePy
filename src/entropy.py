import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from functools import reduce

# breastcancer = pd.read_csv('../results/entropy/BreastCancer.csv', header=0)
# adult = pd.read_csv('../results/entropy/adult.csv', header=0)
# ionosphere = pd.read_csv('../results/entropy/Ionosphere.csv', header=0)
# spambase = pd.read_csv('../results/entropy/spambase.csv', header=0)
# liverdisorder = pd.read_csv('../results/entropy/LiverDisorder.csv', header=0)
# breastcancerwisconsin = pd.read_csv('../results/entropy/BreastCancerWisconsin.csv', header=0)
# biodegradation = pd.read_csv('../results/entropy/biodegradation.csv', header=0)
# winequality = pd.read_csv('../results/entropy/winequality.csv', header=0)
# heartdisease = pd.read_csv('../results/entropy/HeartDisease.csv', header=0)
# mushroom = pd.read_csv('../results/entropy/mushroom.csv', header=0)
#
# df_binary = [breastcancer, adult, ionosphere, spambase, liverdisorder, breastcancerwisconsin, biodegradation,winequality, heartdisease,mushroom]
# final_binary_df = reduce(lambda left,right: pd.merge(left,right,on=['major'],
#                                             how='outer'), df_binary)
#
# print(final_binary_df.columns)
# final_binary_df.to_csv(f'../results/entropy/entropy_results.csv')
# fig = px.line(final_binary_df, x="major", y=final_binary_df.columns[1:11], markers=True)
# fig.update_traces(textposition="bottom right")
# fig.show()


winequality = pd.read_csv('../results/entropy/winequality.csv', header=0)
heartdisease = pd.read_csv('../results/entropy/HeartDisease.csv', header=0)
mushroom = pd.read_csv('../results/entropy/mushroom.csv', header=0)
abalone = pd.read_csv('../results/entropy/Abalone.csv', header=0)
carevaluation = pd.read_csv('../results/entropy/CarEvaluation.csv', header=0)
glassidentification = pd.read_csv('../results/entropy/GlassIdentification.csv', header=0)


df_nonbinary = [winequality, heartdisease, mushroom, abalone, carevaluation,glassidentification]
final_nonbinary_df = reduce(lambda left,right: pd.merge(left,right,on=['major'],
                                            how='outer'), df_nonbinary)
print(final_nonbinary_df.columns)
final_nonbinary_df.to_csv(f'../results/entropy/entropy_results.csv')
fig = px.line(final_nonbinary_df, x="major", y=final_nonbinary_df.columns[1:7], markers=True)
fig.update_traces(textposition="bottom right")
fig.show()
