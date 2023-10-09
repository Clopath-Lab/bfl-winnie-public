""" This is a script we wrote to go around kedro's pipelines till we upgrade
    the other repo to work with the new kedro version or we just move everything
    in here.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append("./src/")
from winnie3.d00_utils.search import qa_process, word_search_state

PATH = Path("/datadrive/prm/messages")


def main():
    primary_messages = pd.read_parquet(PATH)
    qa_dict = qa_process(primary_messages)
    word_search_state(qa_dict)
    print(f"Everything went fine.")


if __name__ == "__main__":
    main()
