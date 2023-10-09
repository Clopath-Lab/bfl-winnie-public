#%%
from sklearn import preprocessing
import torch as th
import pickle
from typing import List
import pandas as pd
from .d00_utils.preprocessing import run_preprocessing_steps


def predict_issues(
    message: str,
    preprocessing_steps: List[str],
    vectorizer_model,
    issue_predict_model,
    labels,
    k: int = 20,
    threshold: float = 0.1,
):
    with th.no_grad():
        preprocessed_text = process_input(message, preprocessing_steps)
        vector = vectorizer_model.transform(preprocessed_text).todense()
        x = th.from_numpy(vector).float()

        top_k = sorted(
            list(
                zip(
                    enumerate(labels),
                    th.sigmoid(issue_predict_model(x)).view(-1).tolist(),
                )
            ),
            key=lambda x: -x[1],
        )

        top_k = [a for a in top_k if a[1] >= threshold][:k]

    all_features = vectorizer_model.get_feature_names_out()
    out_mask = th.LongTensor([a[0][0] for a in top_k])
    feature_importance = get_feature_importance(
        x, out_mask, issue_predict_model, all_features
    )

    return top_k, feature_importance


def case_issues_predict(message: str, train_winnie_settings: dict):

    preprocessing_steps = train_winnie_settings["preprocessing_steps"]
    with open("/datadrive/issues/ch_vectorizer_model", "rb") as f:
        vectorizer_model = pickle.load(f)
    with open("/datadrive/issues/ch_issue_predict_model", "rb") as f:
        issue_predict_model = pickle.load(f)
    with open("/datadrive/issues/ch_issue_predict_labels", "rb") as f:
        labels = pickle.load(f)

    return predict_issues(
        message, preprocessing_steps, vectorizer_model, issue_predict_model, labels
    )


def process_input(input, preprocessing_steps):
    input_series = pd.Series(input)
    input_preprocessed = run_preprocessing_steps(
        series=input_series, steps=preprocessing_steps
    )
    return input_preprocessed


def compute_integrated_gradient(batch_x, batch_blank, model, cols):
    mean_grad = 0
    n = 100
    rows = th.arange(len(batch_x))
    for i in range(1, n + 1):
        x = batch_blank + i / n * (batch_x - batch_blank)
        x.requires_grad = True
        y = model(x)[rows, cols]
        (grad,) = th.autograd.grad(y.sum(), x)
        mean_grad += grad / n

    integrated_gradients = (batch_x - batch_blank) * mean_grad
    return integrated_gradients, mean_grad


def get_feature_importance(x, out_mask, model, all_features):
    xs = x.view(1, -1).repeat(len(out_mask), 1)
    x_b = th.zeros_like(xs)
    integrated_gradient, _ = compute_integrated_gradient(xs, x_b, model, out_mask)

    feature_importance = []
    for _col, grad in enumerate(integrated_gradient):
        (important_feature_indices,) = th.where(grad != 0)
        x_features = [all_features[i] for i in important_feature_indices]
        nonzero_grads = grad[important_feature_indices].abs()
        normalized_importance = nonzero_grads / nonzero_grads.max()

        feature_importance.append(
            sorted(
                list(
                    zip(
                        x_features,
                        normalized_importance.tolist(),
                    )
                ),
                key=lambda x: -x[1],
            )
        )

    return feature_importance


#%%
