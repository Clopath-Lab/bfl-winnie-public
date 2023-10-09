from kedro.pipeline import Pipeline, node

from winnie3.d00_utils.issues_transform import correct_df
from winnie3.d00_utils.message_to_issues import message_to_issues
from winnie3.d04_modelling.case_issues_train import case_issues_train


case_issues_training_pipeline = Pipeline(
    [
        node(
            func=correct_df,
            inputs="case_issues",
            outputs="correct_case_issues",
            name="correct_df",
            tags=["issues", "fetch_data"],
        ),
        # node(
        #     func=message_to_issues,
        #     inputs=["raw_case_sms_messages", "correct_case_issues"],
        #     outputs="sms1_case_issues",
        #     name="sms1_case_issues",
        #     tags=["issues", "fetch_data"],
        # ),
        # node(
        #     func=message_to_issues,
        #     inputs=["unique_sms_messages", "correct_case_issues"],
        #     outputs="sms2_case_issues",
        #     name="sms2_case_issues",
        #     tags=["issues", "fetch_data"],
        # ),
        # node(
        #     func=message_to_issues,
        #     inputs=["raw_case_fb_messages", "correct_case_issues"],
        #     outputs="fm_case_issues",
        #     name="fm_case_issues",
        #     tags=["issues", "fetch_data"],
        # ),
        node(
            func=message_to_issues,
            inputs=["raw_consultation_highlights", "correct_case_issues"],
            outputs="ch_case_issues",
            name="ch_case_issues",
            tags=["issues", "fetch_data"],
        ),
        node(
            func=case_issues_train,
            inputs=["ch_case_issues", "params:train_winnie_settings"],
            outputs=[
                "ch_vectorizer_model",
                "ch_issue_predict_model",
                "ch_issue_predict_labels",
            ],
            name="ch_case_issues_train",
            tags=["issues", "train_model"],
        ),
    ]
)
