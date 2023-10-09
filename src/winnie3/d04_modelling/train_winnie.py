import pandas as pd
import regex as re
from spacy.tokens import DocBin

from winnie3.d00_utils.preprocessing import run_preprocessing_steps
from winnie3.d04_modelling.tf_idf import TFIDF
from winnie3.d04_modelling.spacy_pretrained import SpacyVectorizer
from winnie3.d00_utils.filter_outliers import filter_outliers
from winnie3.d00_utils.language_filter import language_filter


def train_winnie(train_data: pd.DataFrame, train_winnie_settings: dict) -> list:
    """Trains Winnie and returns the three components of Winnie
    :param train_data: A pandas dataframe with question-answer pairs. Should contain columns 'question' and 'answer'
    :param train_winnie_settings: Train hyper-parameter settings from the parameters.yml file
    :return: 1. Trained feature vectorizer, 2. Numeric vectors of the train data, 3. Raw text data of the train data.
    """

    # Reading in the parameters for training
    feature_type = train_winnie_settings["feature_type"]
    preprocessing_steps = train_winnie_settings["preprocessing_steps"]
    train_params = train_winnie_settings["train_params"]

    # Setting the vectorizer
    if feature_type == "tfidf":
        feature_vectorizer = TFIDF()
    else:
        if feature_type == "glove":
            pretrained_model = "en_core_web_md"
        elif feature_type == "use":
            pretrained_model = "en_use_md"
        else:
            print("Chosen vectorizer not implemented, pole!")
        feature_vectorizer = SpacyVectorizer(pretrained_model)

    # Dropping unpopulated question answer pairs
    train_data = train_data.dropna(subset=["question", "answer"])
    train_data.reset_index(drop=True, inplace=True)

    train_data = filter_outliers(data=train_data)

    question = train_data["question"]
    answer = train_data["answer"]

    questions_preprocessed = run_preprocessing_steps(
        series=question, steps=preprocessing_steps
    )

    # remove non english questions
    questions_preprocessed = language_filter(questions_preprocessed)

    # match index
    question = question[questions_preprocessed.index]
    answer = answer[questions_preprocessed.index]

    # reset index
    questions_preprocessed.reset_index(drop=True, inplace=True)
    question.reset_index(drop=True, inplace=True)
    answer.reset_index(drop=True, inplace=True)

    # remove messagemerge from answers
    pattern = re.compile("messagemerge", re.I)
    answer = answer.apply(lambda x: pattern.sub("", x) if x else "")

    # preprocess answers - for spacy search
    answers_preprocessed = run_preprocessing_steps(
        series=answer, steps=preprocessing_steps
    )

    # Fitting and saving the model
    if feature_type == "tfidf":
        feature_vectorizer.fit_model(
            train_data=questions_preprocessed, train_params=train_params
        )
    else:
        feature_vectorizer.fit_model()

    # formatting output
    feature_matrix = feature_vectorizer.generate_features(questions_preprocessed)
    feature_dataframe = pd.DataFrame(feature_matrix, index=questions_preprocessed.index)
    feature_dataframe.columns = feature_dataframe.columns.astype(str)

    # raw text
    text_dataframe = pd.DataFrame({"question": question, "answer": answer})
    # preprocessed text
    preprocessed_text_dataframe = pd.DataFrame(
        {"question": questions_preprocessed, "answer": answers_preprocessed}
    )

    # get docs
    question_docs = feature_vectorizer.generate_features(
        answers_preprocessed, get_vectors=False
    )
    answer_docs = feature_vectorizer.generate_features(
        answers_preprocessed, get_vectors=False
    )

    # save as DocBin for efficiency
    q_doc_bin = DocBin()  # default params
    for q in question_docs:
        q_doc_bin.add(q)
    q_bytes_data = q_doc_bin.to_bytes()
    a_doc_bin = DocBin()  # default params
    for a in answer_docs:
        a_doc_bin.add(a)
    a_bytes_data = a_doc_bin.to_bytes()

    return [
        feature_vectorizer.model,
        feature_dataframe,
        text_dataframe,
        preprocessed_text_dataframe,
        (q_bytes_data, a_bytes_data),
    ]
