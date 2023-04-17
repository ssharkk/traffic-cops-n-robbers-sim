import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
sns.set_theme()
X_LIM=None
Y_LIM=None
TITLE=""

with open("capture_times_grids", "rb") as fp:   # Unpickling
    grid_data = pickle.load(fp)



plt.rcParams['axes.axisbelow'] = True

mpg = sns.load_dataset("mpg")

# sns.catplot(
#     data=mpg, x="cylinders", y="acceleration", hue="weight",
#     native_scale=True, zorder=1
# )
# sns.regplot(
#     data=mpg, x="cylinders", y="acceleration",
#     scatter=False, truncate=False, order=2, color=".2",
# )

# with open("benchmark_data_pulp", "rb") as fp:   # Unpickling
#     pulp_data = pickle.load(fp)
# x = np.array([x for x, y in grid_data])
def process_data(grid_data):
    timeouts_separated = {"edges":[], "strat":[]}
    for line in zip(grid_data["edges"], grid_data["to"], grid_data["strat"]):
        for _ in range(line[1]):
            timeouts_separated["edges"].append(line[0])
            timeouts_separated["strat"].append(line[2])
    print(len(timeouts_separated["edges"]), set(timeouts_separated["edges"]))
    timeouts_separated = pd.DataFrame(timeouts_separated)
    grid_data = grid_data[grid_data["ct"] != 0]

    grid_data.loc[:,"bi"] /= grid_data["bu"]
    grid_data.loc[:,"bi"] /= grid_data["ct"]
    grid_data.loc[:,"bu"] /= grid_data["ct"]
    print("min ct stochastic", min(grid_data[grid_data["strat"]=="stochastic"]["ct"]))
    return grid_data, timeouts_separated

def test_plot_1(grid_data, timeouts_separated):
    FIGSIZE = (6, 3*4)
    HEIGHT_RATIOS = {'height_ratios': [1, 3,2,2]}
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=FIGSIZE, sharex=True, gridspec_kw=HEIGHT_RATIOS)
    g = sns.lineplot(data=grid_data, x="edges", y="ct", hue="strat", ax=ax2, legend=False, zorder=11)
    print(timeouts_separated)
    sns.histplot(data=timeouts_separated, x="edges", hue="strat", ax=ax1, stat="count", fill=True, kde=len(set(timeouts_separated["edges"])) > 1, binwidth=5, multiple="dodge", legend=False)
    # sns.lineplot(data=grid_data, x="edges", y="bu", hue="strat", ax=ax3, legend=False)
    sns.lineplot(data=grid_data[grid_data["strat"] != "optimal unlimited"], x="edges", y="bu", hue="strat", ax=ax3, legend=False)
    sns.lineplot(data=grid_data[grid_data["strat"] != "optimal unlimited"], x="edges", y="bi", hue="strat", ax=ax4, legend=False)
    sns.stripplot(data=grid_data, x="edges", y="ct", hue="strat", ax=ax2, s=3.5,
        marker="$\circ$",
        alpha=0.65,
        native_scale=True, dodge=True, jitter=0.5, zorder=10, legend="full")
    # ax3.set_yscale('symlog')
    sns.move_legend(ax2, "upper center",
        bbox_to_anchor=(.5, 0), ncol=2, title=None, frameon=False,# labels=['Planar Heuristic', 'Minimal Inconvenience', 'Greedy Stochastic', 'Roadblock Saving', 'Optimal (Unlimited Roadblocks)'],
        )
    # leg = g.axes.flat[0].get_legend()
    leg = g.legend()#.get_legend()
    new_title = 'Strategy'
    leg.set_title(new_title)
    new_labels = ['Greedy Stochastic', 'Minimal Inconvenience', 'Planar Heuristic', 'Roadblock Saving', 'Optimal (Unlimited Roadblocks)']
    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(X_LIM)
        ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(visible=True, which='major', color='w', linewidth=1.0, zorder=-2)
        ax.grid(visible=True, which='minor', color='w', linewidth=0.5, zorder=-2)
        ax.set_axisbelow(True)

    # ax2.set_ylim([0, 65])
    ax2.set_ylim(Y_LIM)

    # sns.despine(bottom=True)
    # plt.setp(fig.axes, yticks=[])
    ax1.set_ylabel("Timeouts count")
    ax2.set_ylabel("Capture time (number of turns)")
    ax3.set_ylabel("Roadblocks per turn")
    ax4.set_ylabel("Inconvenience per roadblock")
    ax4.set_xlabel("Edges in the graph")

    # fig.legend(title='Straategy', loc='upper left', labels=['Planar Heuristic', 'Minimal Inconvenience', 'Greedy Stochastic', 'Roadblock Saving', 'Optimal (Unlimited Roadblocks)'],
        # bbox_to_anchor=(.5, 1))
    fig.suptitle(TITLE)
    fig.tight_layout(h_pad=0.2)

def test_plot_1_no_hist(grid_data, timeouts_separated):
    FIGSIZE = (6, 3*4-2)
    HEIGHT_RATIOS = {'height_ratios': [3,2,2]}
    fig, (ax2, ax3, ax4) = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True, gridspec_kw=HEIGHT_RATIOS)
    g = sns.lineplot(data=grid_data, x="edges", y="ct", hue="strat", ax=ax2, legend=False, zorder=11)
    # print(timeouts_separated)
    # sns.histplot(data=timeouts_separated, x="edges", hue="strat", ax=ax1, stat="count", fill=True, kde=len(set(timeouts_separated["edges"])) > 1, binwidth=5, multiple="dodge", legend=False)
    # sns.lineplot(data=grid_data, x="edges", y="bu", hue="strat", ax=ax3, legend=False)
    sns.lineplot(data=grid_data[grid_data["strat"] != "optimal unlimited"], x="edges", y="bu", hue="strat", ax=ax3, legend=False)
    sns.lineplot(data=grid_data[grid_data["strat"] != "optimal unlimited"], x="edges", y="bi", hue="strat", ax=ax4, legend=False)
    sns.stripplot(data=grid_data, x="edges", y="ct", hue="strat", ax=ax2, s=3.5,
        marker="$\circ$",
        alpha=0.65,
        native_scale=True, dodge=True, jitter=0.5, zorder=10, legend="full")
    # ax3.set_yscale('symlog')
    sns.move_legend(ax2, "upper center",
        bbox_to_anchor=(.5, 0), ncol=2, title=None, frameon=False,# labels=['Planar Heuristic', 'Minimal Inconvenience', 'Greedy Stochastic', 'Roadblock Saving', 'Optimal (Unlimited Roadblocks)'],
        )
    # leg = g.axes.flat[0].get_legend()
    leg = g.legend()#.get_legend()
    new_title = 'Strategy'
    leg.set_title(new_title)
    new_labels = ['Greedy Stochastic', 'Minimal Inconvenience', 'Planar Heuristic', 'Roadblock Saving', 'Optimal (Unlimited Roadblocks)']
    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)

    for ax in [ax2, ax3, ax4]:
        ax.set_xlim(X_LIM)
        ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(visible=True, which='major', color='w', linewidth=1.0, zorder=-2)
        ax.grid(visible=True, which='minor', color='w', linewidth=0.5, zorder=-2)
        ax.set_axisbelow(True)

    # ax2.set_ylim([0, 65])
    ax2.set_ylim(Y_LIM)

    # sns.despine(bottom=True)
    # plt.setp(fig.axes, yticks=[])
    # ax1.set_ylabel("Timeouts count")
    ax2.set_ylabel("Capture time (number of turns)")
    ax3.set_ylabel("Roadblocks per turn")
    ax4.set_ylabel("Inconvenience per roadblock")
    ax4.set_xlabel("Edges in the graph")

    # fig.legend(title='Straategy', loc='upper left', labels=['Planar Heuristic', 'Minimal Inconvenience', 'Greedy Stochastic', 'Roadblock Saving', 'Optimal (Unlimited Roadblocks)'],
        # bbox_to_anchor=(.5, 1))
    fig.suptitle(TITLE)
    fig.tight_layout(h_pad=0.2)


def test_plot_2():
    x_vars = ["edges"]
    y_vars = ["ct", "bu", "bi", "tt"]
    x_vars = ["edges","ct", "bu", "bi", "tt"]
    y_vars = ["edges","ct", "bu", "bi", "tt"]
    g = sns.PairGrid(grid_data, hue="strat", x_vars=x_vars, y_vars=y_vars)
    g.map_diag(sns.histplot, color=".3")
    g.map_lower(sns.kdeplot)
    # g.map_offdiag(sns.scatterplot)
    g.add_legend()
    sns.despine(bottom=True)
    # plt.setp(fig.axes, yticks=[])
    # fig.tight_layout(h_pad=2)

def test_plot_3():
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(7, 85), sharex=True, sharey=True)
    # sns.kdeplot(data=grid_data, x="edges", y="ct", hue="strat",levels=15, thresh=.002)
    levels = 12
    thresh = 0.01
    # sns.kdeplot(data=grid_data[grid_data["strat"]=="stochastic"], x="edges", y="ct",levels=levels, thresh=thresh, fill=False, ax=ax1)
    # sns.kdeplot(data=grid_data[grid_data["strat"]=="min inconvenience"], x="edges", y="ct",levels=levels, thresh=thresh, fill=False, ax=ax2)
    # sns.kdeplot(data=grid_data[grid_data["strat"]=="saving"], x="edges", y="ct",levels=levels, thresh=thresh, fill=False, ax=ax3)
    # sns.kdeplot(data=grid_data[grid_data["strat"]=="planar heuristic"], x="edges", y="ct",levels=levels, thresh=thresh, fill=False, ax=ax4)
    # sns.kdeplot(data=grid_data[grid_data["strat"]=="optimal unlimited"], x="edges", y="ct",levels=levels, thresh=thresh, fill=False, ax=ax5)

    sns.kdeplot(data=grid_data[grid_data["strat"]=="stochastic"], x="edges", y="ct",levels=levels, thresh=thresh, fill=False, ax=ax1)
    sns.kdeplot(data=grid_data[grid_data["strat"]=="min inconvenience"], x="edges", y="ct",levels=levels, thresh=thresh, fill=False, ax=ax2)
    sns.kdeplot(data=grid_data[grid_data["strat"]=="saving"], x="edges", y="ct",levels=levels, thresh=thresh, fill=False, ax=ax3)
    sns.kdeplot(data=grid_data[grid_data["strat"]=="planar heuristic"], x="edges", y="ct",levels=levels, thresh=thresh, fill=False, ax=ax4)
    sns.kdeplot(data=grid_data[grid_data["strat"]=="optimal unlimited"], x="edges", y="ct",levels=levels, thresh=thresh, fill=False, ax=ax5)

def test_plot_4(grid_data):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(4, 2, figsize=FIGSIZE, sharex=True, gridspec_kw=HEIGHT_RATIOS)
    sns.histplot(data=timeouts_separated, x="edges", hue="strat", ax=ax1, stat="count", fill=True, kde=True, binwidth=5, multiple="dodge", legend=False)
    sns.lineplot(data=grid_data, x="edges", y="ct", hue="strat", ax=ax2, legend=False)
    sns.lineplot(data=grid_data, x="edges", y="bu", hue="strat", ax=ax3, legend=False)
    sns.lineplot(data=grid_data, x="edges", y="bi", hue="strat", ax=ax4, legend=False)
    sns.stripplot(data=grid_data, x="edges", y="ct", hue="strat", ax=ax2, s=3.5,
        marker="$\circ$",
        alpha=0.65,
        native_scale=True, dodge=True, jitter=0.5, zorder=-1, legend="full")

    sns.histplot(data=timeouts_separated, x="edges", hue="strat", ax=ax1, stat="count", fill=True, kde=True, binwidth=5, multiple="dodge", legend=False)
    sns.lineplot(data=grid_data, x="edges", y="ct", hue="strat", ax=ax2, legend=False)
    sns.lineplot(data=grid_data, x="edges", y="bu", hue="strat", ax=ax3, legend=False)
    sns.lineplot(data=grid_data, x="edges", y="bi", hue="strat", ax=ax4, legend=False)
    sns.stripplot(data=grid_data, x="edges", y="ct", hue="strat", ax=ax2, s=3.5,
    marker="$\circ$",
    alpha=0.65,
    native_scale=True, dodge=True, jitter=0.5, zorder=-1, legend="full")
    sns.move_legend(ax2, "upper left")
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(X_LIM)
    ax2.set_ylim(Y_LIM)
    fig.tight_layout(h_pad=0.2)
    fig.suptitle(TITLE)


# test_plot_4(grid_data) # only for watts_strogatz_sim (4x2)
# test_plot_2()
# test_plot_3()



# plt.scatter([1-x/100 for x, y in minz_data], [y for x, y in minz_data], label="minizinc", s=1.5)
# plt.scatter([1-x/100 for x, y in pulp_data], [y for x, y in pulp_data], label="PuLP", s=1.5)
# plt.scatter([x for x, y in grid_data], [y for x, y in grid_data], label="Grid graph", s=5)
# plt.scatter(minz_data)
# plt.scatter(pulp_data)
# plt.axvline(112)
# plt.ylabel("Capture time (cop turns), timeout=200")
# plt.xlabel("edge count (left: grid with only removed edges, right: grid with only added edges)")
# plt.title("Capture time of (1r, 1c, 1b)-instances on an 8x8 grid; \naverage of greedy strategy runs where cop starts at (0,0) \nand robber tries every other node as starting position")
# plt.scatter(minz_data)
# plt.scatter(pulp_data)


# plt.plot()
# plt.waitforbuttonpress()


X_LIM=None
Y_LIM=None
# grid_data = pd.read_csv("grid_data2.csv")
# SAVE_FNAME = "grid_sim.png"
# Y_LIM=[0,65]
grid_data = pd.read_csv("grid_data3.csv")
SAVE_FNAME = "grid_sim3r.png"
Y_LIM=[0,6]
TITLE = "Performance of strategies\non random graphs based on a square grid (8x8)"
grid_data, timeouts = process_data(grid_data)
test_plot_1(grid_data, timeouts) # (4x1)
# test_plot_1_no_hist(grid_data, timeouts) # (4x1)
plt.savefig(SAVE_FNAME, dpi=250)

X_LIM=None
Y_LIM=None
# grid_data = pd.read_csv("gnm_data2.csv")
# SAVE_FNAME = "gnm_sim.png"
Y_LIM=[0,35]
grid_data = pd.read_csv("gnm_data3.csv")
SAVE_FNAME = "gnm_sim3r.png"
Y_LIM=[0,7]
TITLE = "Performance of strategies\non $\mathregular{G_{n,m}}$ random graphs"
grid_data, timeouts = process_data(grid_data)
test_plot_1(grid_data, timeouts) # (4x1)
# test_plot_1_no_hist(grid_data, timeouts) # (4x1)
plt.savefig(SAVE_FNAME, dpi=250)

X_LIM=None
Y_LIM=None
# grid_data = pd.read_csv("watts_strogatz_data2.csv")
# SAVE_FNAME = "watts_strogatz_sim.png"
grid_data = pd.read_csv("watts_strogatz_data3.csv")
SAVE_FNAME = "watts_strogatz_sim3r.png"
Y_LIM=[0,22]
TITLE = "Performance of strategies\non Newman–Watts–Strogatz small-world graphs"
grid_data, timeouts = process_data(grid_data)
test_plot_1(grid_data, timeouts) # (4x1)
# test_plot_1_no_hist(grid_data, timeouts) # (4x1)
plt.savefig(SAVE_FNAME, dpi=250)

X_LIM=None
Y_LIM=None
# grid_data = pd.read_csv("planar_data2.csv")
# SAVE_FNAME = "planar_sim.png"
# Y_LIM=[0,7]
grid_data = pd.read_csv("planar_data3.csv")
SAVE_FNAME = "planar_sim3r.png"
Y_LIM=[0,6]
TITLE = "Performance of strategies\non randomly generated planar graphs"
grid_data, timeouts = process_data(grid_data)
test_plot_1(grid_data, timeouts) # (4x1)
# test_plot_1_no_hist(grid_data, timeouts) # (4x1)
plt.savefig(SAVE_FNAME, dpi=250)
