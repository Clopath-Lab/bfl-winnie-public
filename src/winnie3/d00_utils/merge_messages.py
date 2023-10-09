import difflib
import pandas as pd
import numpy as np


def merge_messages(messages):
    result = [messages.iloc[0]]
    for j in range(1, len(messages)):
        other_message = messages.iloc[j]
        s = [
            difflib.SequenceMatcher(
                lambda s: not str.isalnum(s), message, other_message
            ).ratio()
            < 0.9
            for message in result
        ]
        if all(s):
            result.append(other_message)
    result = " messagemerge ".join(result)
    return result
