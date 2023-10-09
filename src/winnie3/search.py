import datetime
import lzma
import math
import pickle
import random
import time
from collections import defaultdict
from copy import copy, deepcopy
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from rich import print  # pylint: disable: redefined-builtin
from typing_extensions import Literal

DocId = Tuple[str, int]
DocInfo = Dict[DocId, Dict[str, Any]]
WordList = List[str]
WordScores = Dict[str, float]
Metric = Literal["cosine", "euclidean", "overlap"]


class WordMatch:
    """Here we just build a mapping from words (we assume these are already
            preprocessed) to document ids.
    ​
            It creates a Bag of Words representation. (not if you provide the scores)
    """

    def __init__(self, default_word_score: float = 1.0) -> None:
        self._word_to_docs = dict()  # type: Dict[str, Dict[DocId, float]]
        self._word_scores = dict()  # type: Dict[str, float]
        self._default_word_score = float(default_word_score)
        self._doc_info = dict()  # type: Dict[DocId, Dict[str, Any]]

    def process(
        self, docs: Union[Dict[DocId, WordList], Dict[DocId, WordScores]]
    ) -> None:
        """Here we add to the mapping the given docs. Each document is represented as
        a mapping between words and scores (e.g. term-frequency) or just as a list
        of words (and we assume same score of 1.0 for all words).
        """
        for doc_id, words in docs.items():
            if isinstance(words, list):
                scores = defaultdict(float)  # type: DefaultDict[str, float]
                for word in words:
                    scores[word] += 1.0
            elif isinstance(words, dict):
                scores = words
            else:
                raise ValueError(f"Expected a list or a dict, but got this: {words}")
            for word, score in scores.items():
                self._word_to_docs.setdefault(word, dict())[doc_id] = score

    @property
    def word_to_docs(self) -> Dict[str, Dict[DocId, float]]:
        return self._word_to_docs

    @property
    def word_scores(self) -> WordScores:
        return self._word_scores

    @word_scores.setter
    def word_scores(self, values: WordScores):
        self._word_scores = copy(values)

    @property
    def doc_info(self) -> Dict[DocId, Dict[str, Any]]:
        return self._doc_info

    @doc_info.setter
    def doc_info(self, values: Dict[DocId, Dict[str, Any]]):
        self._doc_info = deepcopy(values)

    def __len__(self) -> int:
        """Returns the vocabulary size."""
        return len(self._word_to_docs)

    def search(
        self,
        nice_to_have: Optional[Tuple[str]] = None,  #  TODO: think about naming
        must_have: Optional[Tuple[str]] = None,
        must_not_have: Optional[Tuple[str]] = None,
        sort: bool = True,
        threshold: float = 0.0,
        top_k: int = -1,
        metric: Metric = "cosine",
        labels: Tuple[str, ...] = ("Q", "A"),
        time_filter: Optional[datetime.datetime] = None,
        max_length: int = 50,
    ) -> List[Tuple[DocId, float, Dict[str, float]]]:
        """Here we return the documents that match the given set of words.
        Set top_k to get at most k results.
        """
        # TODO: make this function a generator for paging (lazy loading)
        doc_scores = defaultdict(float)  # type: DefaultDict[DocId, float]
        doc_match = defaultdict(lambda: defaultdict(float))
        black_list = set()
        if must_not_have:
            for word in must_not_have:
                word = word.lower()
                black_list.update(self._word_to_docs.get(word, dict()).keys())
        scale = 1 / math.sqrt(
            (len(nice_to_have) if nice_to_have else 0)
            + (len(must_have) if must_have else 0)
        )
        if must_have:
            for idx, word in enumerate(must_have):
                word = word.lower()
                score = self._word_scores.get(word, self._default_word_score)
                new_doc_scores = dict()  # type: Dict[DocId, float]
                new_doc_match = defaultdict(lambda: defaultdict(float))
                for doc_id, doc_score in self._word_to_docs.get(word, dict()).items():
                    if doc_id in black_list:
                        continue
                    if idx > 0 and doc_id not in doc_scores:
                        continue
                    if doc_id[0] not in labels:
                        continue
                    if time_filter and self.doc_info[doc_id]["timestamp"] < time_filter:
                        continue
                    crt_term = doc_score * score * scale * score
                    new_doc_scores[doc_id] = doc_scores[doc_id] + crt_term
                    new_doc_match[doc_id] = doc_match[doc_id]
                    new_doc_match[doc_id][word] = crt_term
                doc_scores, doc_match = new_doc_scores, new_doc_match
        if nice_to_have:
            for word in nice_to_have:
                word = word.lower()
                score = self._word_scores.get(word, self._default_word_score)
                for doc_id, doc_score in self._word_to_docs.get(word, dict()).items():
                    if doc_id in black_list:
                        continue
                    if must_have and doc_id not in doc_scores:
                        continue
                    if doc_id[0] not in labels:
                        continue
                    if time_filter and self.doc_info[doc_id]["timestamp"] < time_filter:
                        continue
                    term = doc_score * score * scale * score
                    doc_scores[doc_id] += term
                    doc_match[doc_id][word] = term
        if metric == "cosine":
            for doc_id in doc_scores:
                doc_scores[doc_id] /= self._doc_info[doc_id]["tfidf_norm"]
        elif metric == "euclidean":
            for doc_id in doc_scores:
                doc_scores[doc_id] -= self._doc_info[doc_id]["tfidf_norm"]

        # Filter, sort, and select results
        filtered = list(filter(lambda x: x[1] >= threshold, doc_scores.items()))
        maybe_sorted = sorted(filtered, key=lambda x: -x[1]) if sort else filtered
        results = maybe_sorted if top_k < 1 else maybe_sorted[:top_k]
        # We add the needed details
        return [
            (doc_id, score, doc_match[doc_id])
            for (doc_id, score) in results[:max_length]
        ]

    def load_state(self, path: Union[Path, str]) -> None:
        """Here we load the mappings from the disk, applying LZMA decompression."""
        path = Path(path)
        with lzma.open(path, "rb") as hndlr:
            state = pickle.load(hndlr)
        self._word_to_docs = state["word_to_docs"]
        self._word_scores = state["scores"]
        self._doc_info = state["doc_info"]

    def save_state(self, path: Union[Path, str]) -> None:
        """Here we save the current mappings to disk, applying LZMA compression."""
        path = Path(path)
        if not path.name.endswith(".xz"):
            raise Warning("You should use the .xz extension as we apply lzma.")
        with lzma.open(path, "wb") as hndlr:
            pickle.dump(
                {
                    "word_to_docs": self._word_to_docs,
                    "scores": self._word_scores,
                    "doc_info": self._doc_info,
                },
                hndlr,
            )


def tf_idf(
    docs: Dict[DocId, List[str]]
) -> Tuple[Dict[DocId, WordScores], WordScores, DocInfo]:
    """Takes the documents as lists of strings (words) and returns the same documents
    but wiht TF terms associated to words and also IDF terms in a second dictionary.
    """
    doc_count = defaultdict(float)  # type: DefaultDict[str, float]
    new_docs = {}  # type: Dict[DocId, Dict[str, float]]
    for doc_id, words in docs.items():
        count = defaultdict(float)  # type: DefaultDict[str, float]
        for word in words:
            count[word] += 1.0
        new_words = dict()  # type: Dict[str, float]
        new_docs[doc_id] = new_words
        for word, n in count.items():
            doc_count[word] += 1.0
            new_words[word] = n / math.sqrt(len(words))
    idf = {word: -math.log(k / len(docs)) for (word, k) in doc_count.items()}
    doc_norms = {}
    for doc_id, words in docs.items():
        tf_idf_norm = 0.0
        for word in words:
            tf_idf_norm += (new_docs[doc_id][word] * idf[word]) ** 2
        doc_norms[doc_id] = {"tfidf_norm": math.sqrt(tf_idf_norm)}
    return new_docs, idf, doc_norms


# this needs to be plugged in
def sbert(docs: Dict[DocId, List[str]]) -> Dict[DocId, WordScores]:
    """Takes the documents as lists of (preprocessed) sentences and returns the same documents
    but with sbert embedding.
    """
    embedder = SentenceTransformer(
        "bert-base-nli-mean-tokens"
    )  #  apparently this embedder got discontinued, might need to be changed, see https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
    embedded_sentences = dict()
    for doc_id, sentence in docs.items():
        embedded_sentences[doc_id] = embedder.encode([sentence])
    return embedded_sentences


def print_results(docs, matches):
    for doc_id, score, weights in matches:
        words = docs[doc_id]
        annotated = " ".join(
            [
                (
                    f"[bold]{w} [yellow]({weights[w]:.4f})[/yellow][/bold]"
                    if w in weights
                    else w
                )
                for w in words
            ]
        )
        print(f"[bold][green]{score:7.4f}[/green][/bold] : {annotated}")


def test():
    # Here we have some documents:
    # docs = messages_table["question_tokens"].to_dict()

    docs = docs = {
        ("Q", 1): ["aaa", "bbb", "ccc"],
        ("Q", 2): ["aaa", "ccc", "jjj", "xxx", "yyy", "zzz"],
        ("Q", 3): ["aaa", "ccc", "ddd"],
        ("Q", 4): ["aaa", "eee", "fff", "xxx"],
        ("Q", 5): ["ddd", "ggg", "hhh"],
        ("Q", 6): ["eee", "fff", "iii", "xxx"],
        ("Q", 7): ["ddd", "eee", "fff"],
        ("Q", 8): ["aaa", "ccc", "hhh", "xxx", "yyy"],
    }

    new_docs, idf, doc_info = tf_idf(docs)

    for info in doc_info.values():
        info["timestamp"] = datetime.datetime(2015 + random.randint(0, 6), 3, 15)

    word_match = WordMatch()
    word_match.process(new_docs)
    word_match.word_scores = idf
    word_match.doc_info = doc_info

    # Test dump and load
    word_match.save_state("/datadrive/search/test.xz")
    word_match.load_state("/datadrive/search/test.xz")

    print(f"{len(word_match):d} words in dictionary.")

    # Queries
    while True:
        query = input("Search for (q to quit): ")
        if query == "q":
            break
        parts = list(filter(lambda s: s, set(query.split())))
        must_not_have = [p[1:] for p in parts if p.startswith("-")]
        must_have = [p[1:] for p in parts if p.startswith("+")]
        other = [p for p in parts if not (p.startswith("-") or p.startswith("+"))]
        start = time.time()
        result = word_match.search(other, must_have, must_not_have, top_k=20)
        end = time.time()
        print(f"Found {len(result):d} matches in {(end-start)/1000:.3f} miliseconds.")
        print_results(docs, result)
        print()


if __name__ == "__main__":
    test()
