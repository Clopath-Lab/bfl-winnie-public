import torch
import matplotlib.pyplot as plt
import numpy as np


#%%

colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
embeddings = ["tfidf", "glove", "glove_wmd", "use"]

cs_all = {}
sm_all = {}
wmd_all = {}
cs = {}
sm = {}
wmd = {}


for emb in embeddings:
    cs[emb] = torch.load("/datadrive/eval/cs_" + str(emb) + ".pkl")
    sm[emb] = torch.load("/datadrive/eval/sm_" + str(emb) + ".pkl")
    cs_all[emb] = torch.load("/datadrive/eval/cs_all_" + str(emb) + ".pkl")
    sm_all[emb] = torch.load("/datadrive/eval/sm_all_" + str(emb) + ".pkl")
    try:
        wmd_all[emb] = torch.load("/datadrive/eval/wmd_all_" + str(emb) + ".pkl")
        wmd[emb] = torch.load("/datadrive/eval/wmd_" + str(emb) + ".pkl")
    except:
        continue

top_1_scores = torch.load("/datadrive/eval/top_1_scores.pkl")
top_5_scores = torch.load("/datadrive/eval/top_5_scores.pkl")


x_deltas = [-1, -0.5, 0, +0.5, +1]

with plt.xkcd():
    mosaic = """
        AB
        CD
        """
    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    axd = fig.subplot_mosaic(mosaic)

    axd["A"].set_ylim(-0.05, 1.05)
    axd["A"].sharey(axd["B"])

    for key in axd.keys():
        axd[key].spines["right"].set_visible(False)
        axd[key].spines["top"].set_visible(False)
        axd[key].set_xticks([0.0, 5.0, 10.0])
        axd[key].set_xticklabels(["random", "top-1", "top-5"], rotation=45)

        axd[key].set_xlim(-2, 12)

    x = np.array([0, 5, 10])

    for i, emb in enumerate(embeddings):
        axd["A"].errorbar(
            x=x + x_deltas[i],
            y=[
                cs_all[emb].mean(),
                cs[emb][:, 0].mean(),
                cs[emb][:, :5].max(axis=1).values.mean(),
            ],
            yerr=[
                cs_all[emb].std(),
                cs[emb][:, 0].std(),
                cs[emb][:, :5].max(axis=1).values.std(),
            ],
            ls="none",
            capsize=5,
            color=colors[i],
        )
        axd["A"].scatter(
            x=x + x_deltas[i],
            y=[
                cs_all[emb].mean(),
                cs[emb][:, 0].mean(),
                cs[emb][:, :5].max(axis=1).values.mean(),
            ],
            color=colors[i],
            label=emb,
        )

        axd["B"].errorbar(
            x=x + x_deltas[i],
            y=[
                sm_all[emb].mean(),
                sm[emb][:, 0].mean(),
                sm[emb][:, :5].max(axis=1).values.mean(),
            ],
            yerr=[
                sm_all[emb].std(),
                sm[emb][:, 0].std(),
                sm[emb][:, :5].max(axis=1).values.std(),
            ],
            ls="none",
            capsize=5,
            color=colors[i],
        )
        axd["B"].scatter(
            x=x + x_deltas[i],
            y=[
                sm_all[emb].mean(),
                sm[emb][:, 0].mean(),
                sm[emb][:, :5].max(axis=1).values.mean(),
            ],
            color=colors[i],
            label=emb,
        )

        axd["D"].bar(
            x=x[1:] + x_deltas[i],
            height=[
                top_1_scores[:, i].sum() / len(top_1_scores) * 100,
                top_5_scores[:, i].sum() / len(top_5_scores) * 100,
            ],
            width=0.5,
            color=colors[i],label=emb,
        )

        try:
            axd["C"].errorbar(
                x=x + x_deltas[i],
                y=[
                    wmd_all[emb].mean(),
                    wmd[emb][:, 0].mean(),
                    wmd[emb][:, :5].min(axis=1).values.mean(),
                ],
                yerr=[
                    wmd_all[emb].std(),
                    wmd[emb][:, 0].std(),
                    wmd[emb][:, :5].min(axis=1).values.std(),
                ],
                ls="none",
                capsize=5,
                color=colors[i],
            )
            axd["C"].scatter(
                x=x + x_deltas[i],
                y=[
                    wmd_all[emb].mean(),
                    wmd[emb][:, 0].mean(),
                    wmd[emb][:, :5].min(axis=1).values.mean(),
                ],
                color=colors[i],
                label=emb,
            )

        except:
            continue


axd["A"].set_ylabel("cosine similarity")


axd["B"].set_ylabel("text diff")


axd["C"].set_ylabel("wmd")
axd["C"].invert_yaxis()

axd["D"].set_ylabel("accuracy %")
#axd["D"].set_xlim(3, 12)
axd['D'].set_xticklabels(["", "top-1", "top-5"], rotation=45)
axd["D"].set_ylim(0, 100)
axd["D"].legend()



plt.savefig("/datadrive/eval/all_fig.png", dpi=300)

#%%
