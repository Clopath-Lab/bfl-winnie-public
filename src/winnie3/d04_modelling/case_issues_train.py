import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from winnie3.d00_utils.preprocessing import run_preprocessing_steps
import numpy as np
import torch as th
from torch import nn, optim
from torch.nn import functional as F


def case_issues_train(train_data: pd.DataFrame, train_winnie_settings: dict) -> list:

    # prepare the data

    preprocessing_steps = train_winnie_settings["preprocessing_steps"]
    train_params = train_winnie_settings["train_params"]

    # preprocess
    train_data["pp_message"] = run_preprocessing_steps(
        train_data["message"], steps=preprocessing_steps
    )

    tfidf_model = TfidfVectorizer(**train_params)
    corpus = train_data["pp_message"].values.tolist()
    tfidf_model.fit(corpus)

    feature_matrix = tfidf_model.transform(corpus).todense()
    feature_dataframe = pd.DataFrame(feature_matrix, index=train_data.index)
    feature_dataframe.columns = feature_dataframe.columns.astype(str)

    # prepare tensors

    inputs = th.from_numpy(feature_dataframe.values).float()
    labels = [
        c for c in train_data.columns if c not in {"pp_message", "message", "case_id"}
    ]

    targets = np.vstack(tuple(train_data[c].values for c in labels)).T
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

    # train!

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
            losses.clear()
            evaluate_model(step, tst_inputs, tst_targets)

    return [tfidf_model, model, labels]
