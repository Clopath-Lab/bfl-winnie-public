from kedro.pipeline import Pipeline, node
from winnie3.d02_intermediate.create_intermediate_messages import create_intermediate_messages
from winnie3.d02_intermediate.create_intermediate_sms import create_intermediate_received_sms
from winnie3.d02_intermediate.create_intermediate_sms import create_intermediate_sent_sms


int_pipeline = Pipeline([
    node(
        func=create_intermediate_messages,
        inputs='raw_fb_messages',
        outputs='intermediate_fb_messages',
        name='create_intermediate_messages',
        tags=['int']),
    node(
        func=create_intermediate_received_sms,
        inputs='raw_received_sms',
        outputs='intermediate_received_sms',
        name='create_intermediate_received_sms',
        tags=['int']),
    node(
        func=create_intermediate_sent_sms,
        inputs='raw_sent_sms',
        outputs='intermediate_sent_sms',
        name='create_intermediate_sent_sms',
        tags=['int']),
    ])
