from kedro.pipeline import Pipeline, node
from winnie3.d00_utils.search import qa_process, word_search_state


search_pipeline = Pipeline(
    [
        node(
            func=qa_process,
            inputs="primary_messages",
            outputs="qa_dict",
            name="qa_process",
            tags=["search"],
        ),
        node(
            func=word_search_state,
            inputs="qa_dict",
            outputs=None,
            name="word_search_state",
            tags=["search"],
        ),
    ]
)
