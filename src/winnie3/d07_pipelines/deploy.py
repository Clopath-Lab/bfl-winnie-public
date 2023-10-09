from kedro.pipeline import Pipeline, node
from winnie3.d04_modelling.inference_winnie import make_recommendations
from winnie3.d05_reporting.personalise_recommendations import (
    personalise_recommendations,
)
from winnie3.d02_intermediate.create_intermediate_messages import (
    create_intermediate_messages,
)
from winnie3.d02_intermediate.create_intermediate_sms import (
    create_intermediate_received_sms,
)


def create_deploy_pipeline(case_id, run_winnie, extra_words, search_importance):
    def get_case_id():
        return case_id

    def get_extra_words():
        return extra_words

    def get_run_winnie():
        return run_winnie

    def get_search_importance():
        return search_importance

    deployment_pipeline = Pipeline(
        [
            node(
                func=get_case_id,
                inputs=None,
                outputs="case_id",
                name="get_case_id",
                tags="deploy",
            ),
            node(
                func=get_run_winnie,
                inputs=None,
                outputs="run_winnie",
                name="get_run_winnie",
                tags="deploy",
            ),
            node(
                func=get_extra_words,
                inputs=None,
                outputs="extra_words",
                name="get_extra_words",
                tags="deploy",
            ),
            node(
                func=get_search_importance,
                inputs=None,
                outputs="search_importance",
                name="get_search_importance",
                tags="deploy",
            ),
            node(
                func=make_recommendations,
                inputs=[
                    "case_id",
                    "raw_cases",
                    "model_raw_text",
                    "model_numeric_vectors",
                    "params:deploy_params",
                    "docs_dataframe",
                    "extra_words",
                    "search_importance",
                    "run_winnie",
                ],
                outputs="current_recommendations",
                name="make_recommendations",
                tags="deploy",
            ),
            node(
                func=personalise_recommendations,
                inputs=[
                    "current_recommendations",
                    "case_id",
                    "raw_cases",
                    "intermediate_fb_messages",
                    "intermediate_received_sms",
                ],
                outputs="recommendations",
                name="personalise_recommendations",
                tags="deploy",
            ),
        ]
    )
    return deployment_pipeline
