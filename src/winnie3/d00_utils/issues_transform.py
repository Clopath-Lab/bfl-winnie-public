#%%
from collections import defaultdict
import pickle

with open("/datadrive/issues/issues_transform.pkl", "rb") as handle:
    transform = pickle.load(handle)

#%%


def clean(value):
    return " ".join(value.lower().strip().split())


transform = {k: clean(v) for (k, v) in transform.items()}


def correct_df(case_issues_df):

    case_issues_df["issue"] = case_issues_df.issue.apply(clean)

    print(case_issues_df.issue.nunique(), "original issues")

    # 2. We delete some of the targets
    to_delete = set(k for (k, v) in transform.items() if not v)
    print(f"{len(to_delete)} issues will be deleted.")

    case_issues_df = case_issues_df[~case_issues_df.issue.isin(to_delete)]
    print(f"{case_issues_df.issue.nunique()} unique issues.")

    # 3.
    case_issues_df["issue"] = case_issues_df.issue.apply(lambda v: transform.get(v, v))

    counts = defaultdict(int)

    for _, row in case_issues_df.iterrows():
        counts[row.issue] += 1

    return case_issues_df


# %%
