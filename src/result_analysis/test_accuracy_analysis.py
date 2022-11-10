import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

for file in os.listdir("../../results/accuracies/"):
    print(file)
    result = pd.read_csv("../../results/accuracies/" + file)
    result_major = result[result['minor'] == 0]
    result_major = result_major.drop("minor_train_accuracy", axis=1)
    result_major = result_major.drop("minor", axis=1)
    # print(result_major.head())
    sns.set(rc={"figure.figsize": (15, 10)})
    line_plot = sns.lineplot(data = result_major, x = 'major', y='minor_test_accuracy')
    plt.legend(labels=["Adult",'BreastCancer','CarEvaluation',"HeartDisease","Nursery"])
fig = line_plot.get_figure()
fig.savefig('my_lineplot.png')