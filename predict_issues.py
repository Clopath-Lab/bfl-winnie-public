# %% -- import stuff

from pathlib import Path
from typing import Dict
import pandas as pd
import pickle as pkl
from collections import defaultdict
from src.winnie3.d00_utils.preprocessing import run_preprocessing_steps


# %% -- read datasets

PATH = Path("/datadrive/issues")
FILENAMES = ["ch_case_issues", "fm_case_issues", "sms1_case_issues", "sms2_case_issues"]
PREPROCESS_STEPS = [
    "denoise_text",
    "remove_punctuation",
    "replace_numbers_with_words",
    "remove_punctuation",
    "remove_custom_stop_words",
    "replace_words",
]


def read_datasets() -> Dict[str, pd.DataFrame]:
    datasets = dict()
    for filename in FILENAMES:
        with open(PATH / filename, "rb") as h:
            datasets[filename] = issues_df = pkl.load(h)
            issues_df["text"] = run_preprocessing_steps(
                issues_df["message"], steps=PREPROCESS_STEPS
            )
    return datasets


datasets = read_datasets()
# %%
# import nltk

# nltk.download("brown")
# from nltk.corpus import brown

# brown_words = set(x.lower() for x in brown.words())

# %%
# new_words = defaultdict(int)
# for dataset in datasets.values():
#     for text in dataset["text"].values:
#         for word in text.split():
#             if word not in brown_words:
#                 new_words[word] += 1
# new_words = sorted(new_words.items(), key=lambda x: -x[1])

# %%

# WORD = " dis "
# for dataset in datasets.values():
#     for text in dataset["text"].values:
#         if WORD in text:
#             idx = text.find(WORD)
#             print(text[max(0, idx - 10) : min(len(text), idx + 10)])
# %% -- fun

from sklearn.feature_extraction.text import TfidfVectorizer

# %%
train_data = datasets["ch_case_issues"]

train_params = {"max_df": 0.9, "min_df": 0.01}
model = TfidfVectorizer(**train_params)
corpus = train_data["text"].values.tolist()
model.fit(corpus)

feature_matrix = model.transform(corpus).todense()
feature_dataframe = pd.DataFrame(feature_matrix, index=train_data.index)
feature_dataframe.columns = feature_dataframe.columns.astype(str)


# %%

import numpy as np
import torch as th
from torch import nn, optim
from torch.nn import functional as F


# %% - prepare tensors

dataset = datasets["ch_case_issues"]

inputs = th.from_numpy(feature_dataframe.values).float()

labels = [c for c in dataset.columns if c not in {"text", "message", "case_id"}]

targets = np.vstack(tuple(dataset[c].values for c in labels)).T
targets = th.from_numpy((targets > 0).astype(np.float32)).float()

n = len(inputs)
idxs = th.randperm(n)
ntrain = int(round(0.9 * n))
trn_inputs, tst_inputs = inputs[:ntrain], inputs[ntrain:]
trn_targets, tst_targets = targets[:ntrain], targets[ntrain:]

pos_ratio = (trn_targets > 0).float().mean(dim=0)
neg_ratio = (trn_targets == 0).float().mean(dim=0)
weights = th.where(
    trn_targets > 0,
    (neg_ratio / pos_ratio).repeat(len(trn_targets), 1),
    th.ones_like(trn_targets),
)
weights /= weights.sum(dim=0, keepdim=True)


# %%


model = nn.Sequential(
    nn.Linear(feature_matrix.shape[1], 400),
    nn.Tanh(),
    nn.Linear(400, 300),
    nn.Tanh(),
    nn.Linear(300, len(labels)),
)

optimizer = optim.Adam(model.parameters(), lr=0.001)


@th.no_grad()
def evaluate_model(step, inputs, targets):
    model.eval()
    predictions = model(inputs) > 0.5
    accs = predictions.eq(targets).float().mean(dim=0)
    info = accs.mean().item(), accs.min().item(), accs.max().item()
    print(f"Step {step:3d} | Avg. acc: {info[0] * 100:5.2f}%", end="")

    precision = predictions[targets > 0].float().mean().item()
    print(f" | Precision: {precision * 100:5.2f}%")

    model.train()
    return info


losses = []
for step in range(5000):
    idxs = th.randint(0, len(trn_inputs), (512,))
    logits = model(trn_inputs[idxs])

    loss = F.binary_cross_entropy_with_logits(
        logits, trn_targets[idxs], reduction="none"
    )
    loss = (loss * weights[idxs]).sum(dim=0).mean(dim=0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if step % 10 == 0:
        # print(f"{np.mean(losses):.3f}")
        losses.clear()
        evaluate_model(step, tst_inputs, tst_targets)

# %%
