#%%
import pandas as pd


def message_to_issues(case_messages, case_issues):
    case_messages = case_messages.sort_values("message_date")
    case_messages["message"] = case_messages.groupby(["case_id"])["message"].transform(
        lambda x: " ".join(x)
    )

    case_messages = case_messages.drop(columns="message_date").drop_duplicates()

    case_issues["is_general_enquiry"] = case_issues["issue"].apply(
        lambda x: x.lower() == "general inquiry"
    )

    case_issues_messages = pd.merge(
        case_messages, case_issues, on="case_id", how="inner", validate="one_to_many"
    )

    case_issues_messages = (
        pd.concat(
            [
                case_issues_messages.drop("issue", 1),
                pd.get_dummies(case_issues_messages.issue).mul(1),
            ],
            axis=1,
        )
        .groupby(["case_id", "message"])
        .max()
    )

    case_issues_messages.reset_index(inplace=True)

    return case_issues_messages
