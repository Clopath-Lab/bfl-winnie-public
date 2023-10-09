#%%
# from d00_utils.preprocessing import run_preprocessing_steps
import numpy as np
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import DocBin


#%%
class SpacyMatch:
    def __init__(self, feature_type):
        if feature_type == "glove":
            pretrained_model = "en_core_web_md"
        elif feature_type == "use":
            pretrained_model = "en_use_md"
        self._nlp = spacy.load(pretrained_model)

    def search(self, docs, extra_words=[]):
        doc_bin = DocBin().from_bytes(docs)
        docs = list(doc_bin.get_docs(self._nlp.vocab))
        patterns = [self._nlp.make_doc(x.lower()) for x in extra_words]
        matcher = PhraseMatcher(self._nlp.vocab)
        for idx, p in enumerate(patterns):
            matcher.add(extra_words[idx], [p])
        match_scores = np.array([self.get_matches(matcher, doc) for doc in docs])
        return match_scores

    def get_matches(self, matcher, doc):
        matches = matcher(doc)
        if matches:
            words_matched = []
            for m in matches:
                match_id, start, end = m
                words_matched.append(self._nlp.vocab.strings[match_id])
            words_matched_set = list(set(words_matched))
            word_counts = []
            for word in words_matched_set:
                word_counts.append(words_matched.count(word))
            word_counts = np.array(word_counts)
            match_score = 2 * (
                len(word_counts) + np.log(word_counts[word_counts > 1]).sum()
            )
        else:
            match_score = 1.0
        return match_score


#%%
