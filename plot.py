import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir

# global style
sns.set(style="white", font_scale=1.5, palette="pastel")

def render(outpath):
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved figure to {outpath}")

def plot_utility_vs_nalt(df, by_seed=False, x="n_alt", metric="utility", outpath="./drawings/utility_raw.png"):
    xmax = df[df.agent.str.contains("L_alt")][x].max()
    kwargs = dict(x=x, y=metric, hue="AG_type", style="AG_type")
    XLABELS = {"n_alt": "# alternatives", "prop_alt": "(# alternatives) / $|U|$"}
    YLABELS = {"utility": "utility", "utility_ratio": "utility ratio $(S_1, L_{alt})/(S_1,L_1)$"}

    if by_seed:
        for i, seed in enumerate(df.seed.unique()):
            sub = df[df.seed==seed]
            legend = (i==0) # only keep legend for first seed, to avoid duplication
            ax = sns.lineplot(data=sub[sub.agent.str.contains("L_alt")], legend=legend, alpha=0.5, **kwargs)
    else:
        ax = sns.lineplot(data=df[df.agent.str.contains("L_alt")], **kwargs)

    # add horizontal lines for S0-L0, S1-L0, and S1-L1
    for agent in ["S0-L0", "S1-L0", "S1-L1"]:
        mean = df[df.agent==agent][metric].mean()
        ax.axhline(mean, linestyle="--", color="k", alpha=0.5)
        ax.text(xmax - (xmax*0.1), mean+0.003, agent, size="x-small")

    # edit labels and save
    ax.legend(title="", loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=len(df.AG_type.unique()))
    ax.set_xlabel(XLABELS[x])
    ax.set_ylabel(YLABELS[metric])
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
    df["prop_alt"] = df.n_alt / df.n_alt.max()
    df.to_csv("./data/utility.csv", index=False)

    # check if any models actually do better than (S1, L1)
    print("models better than (S1, L1):")
    from numpy import isclose
    print(df[(df.utility_ratio < 1)&~(isclose(df.utility_ratio, 1))])

    # generate plots
    plot_utility_vs_nalt(df, metric="utility", outpath="./drawings/utility_raw.png")
    plt.clf()
    plot_utility_vs_nalt(df, metric="utility_ratio", outpath="./drawings/utility_ratio.png")