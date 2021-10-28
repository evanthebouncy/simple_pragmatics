import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# global style
sns.set(style="white", font_scale=1.5, palette="pastel")

def render(outpath):
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved figure to {outpath}")

def plot_training_curve(df, metric="loss", outpath="./drawings/blackbox_training_loss.png"):
    sns.lineplot(data=df, x="step", y=metric)
    render(outpath)

if __name__ == "__main__":
    df = pd.read_csv("./debug/train-loss.csv")
    plot_training_curve(df, metric="loss", outpath="./drawings/blackbox_training_loss.png")
    plt.clf()
    plot_training_curve(df, metric="utility", outpath="./drawings/blackbox_training_utility.png")
