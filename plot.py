import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir

# global style
sns.set(style="white", font_scale=1.5, palette="pastel")

def render(outpath):
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved figure to {outpath}")

def plot_utility_vs_nalt(df, by_seed=False, metric="utility", outpath="./drawings/utility_raw.png"):
    nU = df.n_alt.max()
    kwargs = dict(x="n_alt", y=metric, hue="AG_type")

    if by_seed:
        for seed in df.seed.unique():
            sub = df[df.seed==seed]
            ax = sns.lineplot(data=sub[sub.agent.str.contains("L_alt")], legend=False, **kwargs)
    else:
        ax = sns.lineplot(data=df[df.agent.str.contains("L_alt")], **kwargs)

    # add horizontal lines for S0-L0, S1-L0, and S1-L1
    for agent in ["S0-L0", "S1-L0", "S1-L1"]:
        mean = df[df.agent==agent][metric].mean()
        ax.axhline(mean, linestyle="--", color="k", alpha=0.5)
        ax.text(nU - (nU*0.1), mean+0.003, agent, size="x-small")

    # edit labels and save
    ax.set_xlabel("# alternatives")
    ax.set_ylabel(metric)
    render(outpath)

if __name__ == "__main__":
    # gather data
    all_dfs = []
    AG_types = ["greedy", "random", "sequential"]
    for AG_type in AG_types:
        dfs = [pd.read_csv(f"./data/{AG_type}/{f}") for f in listdir(f"./data/{AG_type}")]
        for df in dfs:
            df["AG_type"] = AG_type
        all_dfs += dfs
    df = pd.concat(all_dfs)
    df.to_csv("./data/utility.csv", index=False)

    # generate plots
    plot_utility_vs_nalt(df, metric="utility", outpath="./drawings/utility_raw.png")
    plt.clf()
    plot_utility_vs_nalt(df, metric="utility_ratio", outpath="./drawings/utility_ratio.png")