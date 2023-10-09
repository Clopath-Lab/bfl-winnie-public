from functools import partial
from bs4 import BeautifulSoup
import re
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import contractions
from nltk.corpus import stopwords
import string
import num2words
import nltk
import pandas as pd
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import warnings
import pickle


#%%
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")


def drop_shorter(text, length=20):
    return text if len(text) >= length else ""


def clean_blasts(text):
    with open("/datadrive/preprocess/blasts.pkl", "rb") as handle:
        blasts = pickle.load(handle)
    spam_pattern = "|".join(re.compile(x).pattern for x in blasts)
    pattern = re.compile(spam_pattern, re.I)
    if bool(pattern.search(text.lower())):
        return ""
    return text


def strip_html_tags(text):
    """removes HTML tags in the text"""
    soup = BeautifulSoup(text, "html.parser").text
    return soup


def strip_urls(text):
    """Strips any URLs in the text (has to be prepended by http/www)"""
    return re.sub(r"(http|www)\S+", "", text)


def strip_emails(text):
    return re.sub(r"\S*@\S*\s?", "", text)


def remove_between_square_brackets(text):
    """Removes any text placed between square brackets"""
    return re.sub(r"\[[^]]*\]", "", text)


def denoise_text(text):
    """combines all the denoising steps"""
    text = strip_html_tags(text)
    text = strip_urls(text)
    text = strip_emails(text)
    text = remove_between_square_brackets(text)

    return text


def stem_words(text):
    """combines the different forms of the verbs/adverbs/adjectives"""
    text = text.split()
    try:
        stemmer = LancasterStemmer()
    except LookupError:
        nltk.download("wordnet")

    stems = list()
    for word in text:
        stem = stemmer.stem(word)
        stems.append(stem)
    return " ".join(stems)


def lemmatize_words(text):
    """converts the word into its root form"""
    try:
        lemmatizer = WordNetLemmatizer()
    except LookupError:
        nltk.download("wordnet")
        lemmatizer = WordNetLemmatizer()

    lemmas = list()
    for word in text.split():
        lemma = lemmatizer.lemmatize(word, pos="v")
        lemmas.append(lemma)

    return " ".join(lemmas)


def remove_stop_words(text):
    """Remove stop words from raw text"""
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))

    return " ".join([item for item in text.split() if item not in stop_words])


def remove_punctuation(text):
    """Remove all punctuation in text and replace with whitespace"""
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return text.translate(translator)


def replace_numbers_with_words(text):
    """convert numbers into words: e.g. 1 to one"""
    return " ".join(
        [num2words.num2words(s) if s.isdigit() else s for s in text.split()]
    )


def remove_numbers(text):
    """replace numbers with whitespace"""
    return " ".join(["" if s.isdigit() else s for s in text.split()])


def filter_pos_tags(text, list_pos_tags=None):
    """retains the defined POS tags in text and removes the rest"""
    if list_pos_tags is None:
        list_pos_tags = [
            "NN",
            "NNP",
            "NNS",
            "NNPS",
            "VB",
            "VBD" "VBG",
            "VBN",
            "VBP",
            "VBZ",
        ]

    list_retained = [
        word for word, pos in pos_tag(word_tokenize(text)) if pos in list_pos_tags
    ]

    return " ".join(list_retained)


def replace_words(text):
    """correcting some common spelling mistakes / abbrivations"""
    with open("/datadrive/preprocess/word_replacements.pkl", "rb") as handle:
        word_replacements = pickle.load(handle)
    word_replacements = {
        r"\b" + k.strip() + r"\b": v.strip() for (k, v) in word_replacements.items()
    }
    stop_word_pattern = "|".join(
        re.compile(x).pattern for x in word_replacements.keys()
    )
    pattern = re.compile(stop_word_pattern, re.I)
    cleaned_text = re.sub(
        pattern, lambda m: word_replacements[r"\b" + m.group(0) + r"\b"], text.lower()
    )

    return cleaned_text


def remove_custom_stop_words(text):
    """removing words from the text that occur very often in data that don't carry much info"""
    with open("/datadrive/preprocess/custom_stop_words.pkl", "rb") as handle:
        custom_stop_words = pickle.load(handle)
    custom_stop_words = [r"\b" + k.strip() + r"\b" for k in custom_stop_words]
    stop_word_pattern = "|".join(re.compile(x).pattern for x in custom_stop_words)
    pattern = re.compile(stop_word_pattern, re.I)
    cleaned_text = pattern.sub("", text.lower())

    """received SMS messages start with LAW keyword"""
    """remove SMS first keyword requirement."""
    split_text = cleaned_text.lower().split()
    if (len(split_text) > 0) and (
        split_text[0] == "law"
    ):  # does not account for multiple messages from same speaker, each prefixed with 'LAW'
        join_text = " ".join(split_text[1:])
    else:
        join_text = " ".join(split_text)

    return join_text.strip()


def remove_emoji(text):
    regrex_pattern = re.compile(
        pattern="["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        "]+",
        flags=re.UNICODE,
    )
    return regrex_pattern.sub(r"", text)


def run_preprocessing_steps(series: pd.Series, steps: list = None) -> pd.Series:
    """Runs the preprocesing steps sequentially and returns the preprocessed series"""

    pre_process_dict = {
        "remove_punctuation": remove_punctuation,
        "denoise_text": denoise_text,
        "remove_numbers": remove_numbers,
        "replace_contractions": contractions.fix,
        "remove_stop_words": remove_stop_words,
        "stem_words": stem_words,
        "lemmatize_words": lemmatize_words,
        "filter_pos_tags": filter_pos_tags,
        "replace_numbers_with_words": replace_numbers_with_words,
        "remove_custom_stop_words": remove_custom_stop_words,
        "remove_emoji": remove_emoji,
        "replace_words": replace_words,
        "clean_blasts": clean_blasts,
        "drop_shorter": drop_shorter,
    }

    if steps is None:
        steps = [
            "denoise_text",
            "remove_punctuation",
            "replace_numbers_with_words",
            "remove_custom_stop_words",
        ]

    # Converting all the entries in the series to lower case.
    series = series.str.lower()

    for step in steps:
        if isinstance(step, (str,)):
            func = pre_process_dict[step]
        elif isinstance(step, (tuple, list)):
            fname, kwargs = step
            func = partial(pre_process_dict[fname], **kwargs)
        else:
            raise ValueError("Expected Tuple[str, Dict[str, Any]] or str.")
        series = series.apply(func)

    return series
