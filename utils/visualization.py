import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(df, columns):
    for column in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

def plot_pairplot(df):
    sns.pairplot(df)
    plt.show()
