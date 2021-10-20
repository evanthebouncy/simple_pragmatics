import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_acc_vs_nalt(df, outpath="./drawings/acc.png"):
    sns.set(style="white", font_scale=1.5, palette="pastel")

    # ax = sns.scatterplot(data=df[df.agent.str.contains("L_alt")], x="n_alt", y="acc", alpha=0.5)
    ax = sns.lineplot(data=df[df.agent.str.contains("L_alt")], x="n_alt", y="acc", hue="AG_type")

    # add horizontal lines for S0-L0, S1-L0, and S1-L1
    for agent in ["S0-L0", "S1-L0", "S1-L1"]:
        mean_acc = df[df.agent==agent].acc.mean()
        ax.axhline(mean_acc, linestyle="--", color="k", alpha=0.5)
        ax.text(115, mean_acc+0.003, agent, size="x-small")

    # edit labels and save
    ax.set_xlabel("# alternatives")
    ax.set_ylabel("communication accuracy")
    plt.savefig(outpath, bbox_inches="tight")

if __name__ == "__main__":
    df_seq = pd.read_csv("./drawings/acc_sequential.csv")
    df_rand = pd.read_csv("./drawings/acc_random.csv")

    df_seq["AG_type"] = "sequential"
    df_rand["AG_type"] = "random"

    df = pd.concat([df_seq, df_rand])
    plot_acc_vs_nalt(df, outpath="./drawings/acc_combined.png")