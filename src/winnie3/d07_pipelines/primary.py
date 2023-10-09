from kedro.pipeline import Pipeline, node
from winnie3.d03_primary.create_question_answer_fb import (
    create_primary_fb_conversations,
)
from winnie3.d03_primary.create_primary_sms import create_primary_sms
from winnie3.d03_primary.create_primary_messages import create_primary_messages


prm_pipeline = Pipeline(
    [
        node(
            func=create_primary_fb_conversations,
            inputs="intermediate_fb_messages",
            outputs="primary_fb_conversations",
            name="create_primary_fb_conversations",
            tags=["prm"],
        ),
        node(
            func=create_primary_sms,
            inputs=["intermediate_received_sms", "intermediate_sent_sms"],
            outputs="primary_sms",
            name="create_primary_sms",
            tags=["prm"],
        ),
        node(
            func=create_primary_messages,
            inputs=["primary_sms", "primary_fb_conversations"],
            outputs="primary_messages",
            name="create_primary_messages",
            tags=["prm"],
        ),
    ]
)
