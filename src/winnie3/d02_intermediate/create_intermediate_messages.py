import numpy as np
from winnie3.d00_utils.preprocessing import clean_blasts


def create_intermediate_messages(raw_messages):

    intermediate_fb_messages = raw_messages.copy()
    intermediate_fb_messages.drop(
        columns=[
            "user_id",
            "created_at",
            "updated_at",
            "deleted_at",
            "message_type",
            "facebook_message_id",
        ],
        inplace=True,
    )
    intermediate_fb_messages["body"] = intermediate_fb_messages["body"].apply(
        lambda x: clean_blasts(x)
    )
    intermediate_fb_messages["body"].replace("", np.nan, inplace=True)
    intermediate_fb_messages.dropna(subset=["body"], inplace=True)

    return intermediate_fb_messages
