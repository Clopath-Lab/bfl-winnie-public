from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import re
import difflib

from typing import (
    Dict,
    List,
    Optional,
)

from winnie3.d00_utils.preprocessing import run_preprocessing_steps
from winnie3.d00_utils.yaml_utils import get_path_catalog
from winnie3.d00_utils.yaml_utils import get_parameter_yaml
from winnie3.d04_modelling.tf_idf import TFIDF
from winnie3.d04_modelling.spacy_pretrained import SpacyVectorizer
from winnie3.spacy_search import SpacyMatch


def make_recommendations(
    case_id: int,
    raw_cases: pd.DataFrame,
    model_raw_text: pd.DataFrame,
    model_numeric_vectors: pd.DataFrame,
    deploy_params: Dict,
    docs_dataframe: pd.DataFrame,
    extra_words: Optional[str] = [],  # comma separated
    search_importance: Optional[float] = 1.0,
    run_winnie: Optional[bool] = True,
):
    train_winnie_settings = get_parameter_yaml("train_winnie_settings")
    feature_type = train_winnie_settings["feature_type"]
    distance_metric = deploy_params.get("distance_metric")
    n_answers_displayed = deploy_params.get("num_neighbours")
    num_neighbors = len(model_raw_text)

    if not extra_words and not run_winnie:
        print("You did not pass any data!")
        recommendations = pd.DataFrame(
            {
                "response_rank": [0],
                "score": [0],
                "index": [0],
                "recommended_response": [""],
                "case_id": [case_id],
            }
        )
    else:

        # prepare case_data
        current_case_data = raw_cases.loc[raw_cases.id == case_id]
        if run_winnie:
            recommendations = inference_winnie(
                question_data=current_case_data,
                model_raw_text=model_raw_text,
                model_numeric_vectors=model_numeric_vectors,
                feature_type=feature_type,
                distance_metric=distance_metric,
                num_neighbors=num_neighbors,
            )

        if extra_words:
            match_score_combined = winnie_search(
                extra_words, docs_dataframe, feature_type
            )
            if run_winnie and extra_words:
                # adjust scores
                recommendations["score"] = recommendations["score"] / (
                    search_importance * match_score_combined[recommendations["index"]]
                )
                # sort by new scores
                recommendations.sort_values("score", inplace=True, ascending=True)
            else:  # winnie was not run! i.e. not using case data at all.
                # create a recommendations df
                sorted_indices = np.argsort(1 / match_score_combined)
                for question_counter in range(len(current_case_data)):
                    recommendations = pd.DataFrame(
                        {
                            "response_rank": list(np.arange(1, num_neighbors + 1)),
                            "score": match_score_combined[sorted_indices],
                            "index": sorted_indices,
                            "recommended_response": (
                                model_raw_text["answer"].iloc[sorted_indices].tolist()
                            ),
                            "case_id": current_case_data["id"].iloc[question_counter],
                        }
                    )

        distinct_indices = select_k_distinct(recommendations, n_answers_displayed)
        recommendations = recommendations.iloc[distinct_indices]

    recommendations = recommendations.drop(["score", "index"], axis=1)

    return recommendations


def select_k_distinct(r, n_answers_displayed):
    result = [r["recommended_response"].iloc[0]]
    indices = [0]
    j = 0
    while len(result) < n_answers_displayed:
        j += 1
        other_answer = r["recommended_response"].iloc[j]
        s = [
            difflib.SequenceMatcher(
                lambda s: not str.isalnum(s), answer, other_answer
            ).ratio()
            < 0.9
            for answer in result
        ]
        if all(s):
            result.append(other_answer)
            indices.append(j)
    return indices


def winnie_search(extra_words: str, docs_dataframe: pd.DataFrame, feature_type: str):
    spacymatch = SpacyMatch(feature_type)

    extra_words_list = list(filter(lambda s: s, re.split(",", extra_words)))

    # search both Q & A

    match_score_q = spacymatch.search(docs_dataframe[0], extra_words=extra_words_list)
    match_score_a = spacymatch.search(docs_dataframe[1], extra_words=extra_words_list)

    match_score_combined = match_score_q + match_score_a

    return match_score_combined


def inference_winnie(
    question_data: pd.DataFrame,
    model_raw_text: pd.DataFrame,
    model_numeric_vectors: pd.DataFrame,
    feature_type="glove",
    distance_metric="cosine",
    num_neighbors=5,
):
    """Estimated and returns a set of candidate responses to a new question(s)
    :param question_data: Text data and meta data for the new question
    :param model_raw_text: raw text from the train model step
    :param model_numeric_vectors: numeric representation of the questions from the train model step
    :param feature_type: Type of features created, has to match the saved features
    :param distance_metric: distance metric used in the nearest neighbor method
    :param num_neighbors: Number of candidate responses to return
    :return: a pandas dataframe with candidate responses
    """

    # Loading Saved Model files
    saved_model_path = get_path_catalog("trained_model")

    # prepare question for inference
    train_winnie_settings = get_parameter_yaml("train_winnie_settings")
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

    # Note: this is the column name used in the MySQL database.
    question_series = question_data["consultation_highlights"]

    preprocessing_steps = train_winnie_settings["preprocessing_steps"]
    questions_preprocessed = run_preprocessing_steps(
        series=question_series, steps=preprocessing_steps
    )

    questions_preprocessed_features = feature_vectorizer.generate_features(
        intermediate_series=questions_preprocessed, saved_model_path=saved_model_path
    )

    # Set up Nearest Neighbors
    nearest_neighbor_model = NearestNeighbors(
        n_neighbors=num_neighbors,
        metric=distance_metric,  # all potential answers!
    )
    nearest_neighbor_model.fit(model_numeric_vectors)

    # Finding the nearest neighbors
    dist, ind = nearest_neighbor_model.kneighbors(questions_preprocessed_features)

    result = []
    for question_counter in range(len(question_data)):
        recommendation_df = pd.DataFrame(
            {
                "response_rank": list(np.arange(1, num_neighbors + 1)),
                "score": dist[question_counter],  # needed for search bumping
                "index": ind[question_counter],  # needed for search bumping
                "recommended_response": (
                    model_raw_text["answer"].iloc[ind[question_counter]].tolist()
                ),
                "case_id": question_data["id"].iloc[question_counter],
            }
        )
        result.append(recommendation_df)

    result = pd.concat(result, ignore_index=True)

    return result
