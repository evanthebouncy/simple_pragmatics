import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# global style
sns.set(style="white", font_scale=1.5, palette="pastel")

def render(outpath):
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved figure to {outpath}")

def plot_training_curve(df, metric="loss", 
                        outpath="./drawings/blackbox_training_loss.png"):
    sns.lineplot(data=df, x="step", y=metric, units="seed", estimator=None, alpha=0.3)
    render(outpath)

def plot_fn_nalt(df, outpath):
    sns.lineplot(data=df, x="step", y="loss", hue="n_prev_alt") #, legend=False)
    render(outpath)

def plot_utility(df, outpath):
    sns.lineplot(data=df, x="n_prev_alt", y="utility", hue="AG_type", legend="full")
    render(outpath)

def plot_propagree(df, outpath):
    sns.lineplot(data=df, x="n_prev_alt", y="prop_agree_greedy", hue="AG_type", legend="full")
    render(outpath)

if __name__ == "__main__":
    figs = "./drawings/imitate_greedy"
    nseeds = 8
    prefix = "runs/test"
    df = pd.concat([pd.read_csv(f"./{prefix}_seed{seed}/train-loss.csv") for seed in range(nseeds)])
    plot_training_curve(df, metric="loss", outpath=f"{figs}/training_loss.png")
    plt.clf()
    # plot_training_curve(df, metric="utility", outpath="./debug_new/training_utility.png")
    # plt.clf()
    # plot_fn_nalt(df, f"{figs}/training_loss_by_nalt.png")
    # plt.clf()
    df = pd.concat([pd.read_csv(f"./{prefix}_seed{seed}/test-utility.csv") for seed in range(nseeds)])
    plot_utility(df, f"{figs}/test_utility.png")
    plt.clf()
    plot_propagree(df, f"{figs}/test_propagree.png")