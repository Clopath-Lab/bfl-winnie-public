import lzma
import pickle
from datetime import date
import re, collections
from functools import reduce
import nltk
import pandas as pd

from winnie3.d00_utils.preprocessing import run_preprocessing_steps
from winnie3.search import WordMatch, tf_idf


def qa_process(primary_messages):
    # where tokenizer is either word or subword
    q = primary_messages[["id", "question", "question_asked_time"]]
    a = primary_messages[["id", "answer", "answer_given_time"]]
    q = q.dropna()
    a = a.dropna()
    q.columns = ["id", "text", "timestamp"]
    a.columns = ["id", "text", "timestamp"]
    qa = pd.concat([q, a])
    preprocessing_steps = [
        "denoise_text",
        "remove_punctuation",
        "replace_contractions",
        "replace_numbers_with_words",
        "remove_custom_stop_words",
    ]
    qa_preprocessed = run_preprocessing_steps(
        series=qa["text"], steps=preprocessing_steps
    )

    word_qa_tokens = [nltk.word_tokenize(x) for x in qa_preprocessed]

    vocab = get_vocab(reduce(lambda a, b: a + b, word_qa_tokens))

    num_merges = 10000  # hard coded for now
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)

    tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)

    sorted_tokens_tuple = sorted(
        tokens_frequencies.items(),
        key=lambda item: (measure_token_length(item[0]), item[1]),
        reverse=True,
    )
    sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]

    subword_qa_tokens = [
        [tokenize_word(string=word, sorted_tokens=sorted_tokens) for word in sentence]
        for sentence in word_qa_tokens
    ]

    # create the dictionary
    q_ids = [("Q", x) for x in q["id"]]
    a_ids = [("A", x) for x in a["id"]]
    ids = q_ids + a_ids
    qa_dict_word_tokens = dict(zip(ids, word_qa_tokens))
    qa_dict_subword_tokens = dict(zip(ids, subword_qa_tokens))
    qa_dict_preprocessed = dict(zip(ids, qa_preprocessed))
    qa_dict_raw = dict(zip(ids, qa["text"]))
    qa_dict_timestamps = dict(zip(ids, qa["timestamp"]))
    path = "/datadrive/search/qa_dict.xz"
    qa_dict = {
        "qa_dict_word_tokens": qa_dict_word_tokens,
        "qa_dict_subword_tokens": qa_dict_subword_tokens,
        "qa_dict_preprocessed": qa_dict_preprocessed,  #  for sbert!
        "qa_dict_raw": qa_dict_raw,
        "time_stamps": qa_dict_timestamps,
    }
    with lzma.open(path, "wb") as hndlr:
        pickle.dump(qa_dict, hndlr)

    return qa_dict


def word_search_state(qa_dict) -> None:
    new_docs, idf, doc_info = tf_idf(qa_dict["qa_dict_word_tokens"])
    for doc_id, timestamp in qa_dict["time_stamps"].items():
        doc_info[doc_id]["timestamp"] = timestamp.date()
    word_match = WordMatch()
    word_match.process(new_docs)
    word_match.word_scores = idf
    word_match.doc_info = doc_info
    word_match.save_state("/datadrive/search/state.xz")


def get_vocab(words):
    vocab = collections.defaultdict(int)
    for word in words:
        vocab[" ".join(list(word)) + " </w>"] += 1
    return vocab


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word in v_in:
        w_out = p.sub("".join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def get_tokens_from_vocab(vocab):
    tokens_frequencies = collections.defaultdict(int)
    vocab_tokenization = {}
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens_frequencies[token] += freq
        vocab_tokenization["".join(word_tokens)] = word_tokens
    return tokens_frequencies, vocab_tokenization


def measure_token_length(token):
    if token[-4:] == "</w>":
        return len(token[:-4]) + 1
    else:
        return len(token)


def tokenize_word(string, sorted_tokens, unknown_token="</u>"):

    if string == "":
        return []
    if sorted_tokens == []:
        return [unknown_token]

    string_tokens = []
    for i in range(len(sorted_tokens)):
        token = sorted_tokens[i]
        token_reg = re.escape(token.replace(".", "[.]"))

        matched_positions = [
            (m.start(0), m.end(0)) for m in re.finditer(token_reg, string)
        ]
        if len(matched_positions) == 0:
            continue
        substring_end_positions = [
            matched_position[0] for matched_position in matched_positions
        ]

        substring_start_position = 0
        for substring_end_position in substring_end_positions:
            substring = string[substring_start_position:substring_end_position]
            string_tokens += tokenize_word(
                string=substring,
                sorted_tokens=sorted_tokens[i + 1 :],
                unknown_token=unknown_token,
            )
            string_tokens += [token]
            substring_start_position = substring_end_position + len(token)
        remaining_substring = string[substring_start_position:]
        string_tokens += tokenize_word(
            string=remaining_substring,
            sorted_tokens=sorted_tokens[i + 1 :],
            unknown_token=unknown_token,
        )
        break
    return string_tokens
