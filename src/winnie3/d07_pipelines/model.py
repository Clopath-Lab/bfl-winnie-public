from kedro.pipeline import Pipeline, node
from winnie3.d04_modelling.train_winnie import train_winnie


# Defining the nodes for each model type
# User can choose the node(s) to include in the pipeline

train_pipeline = Pipeline(
    [
        node(
            func=train_winnie,
            inputs=["primary_messages", "params:train_winnie_settings"],
            outputs=[
                "trained_model",
                "model_numeric_vectors",
                "model_raw_text",
                "model_preprocessed_text",
                "docs_dataframe",
            ],
            name="train_winnie",
            tags=["model"],
        )
    ]
)
