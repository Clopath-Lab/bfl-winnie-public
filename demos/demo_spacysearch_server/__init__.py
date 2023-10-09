import os
import re
import lzma
from copy import copy
from pathlib import Path
import pickle
from flask import Flask
from flask import render_template
from flask import request
from datetime import date
import nltk
from pandas import Series
import pandas as pd

import sqlalchemy

from .winnie3.d00_utils.preprocessing import run_preprocessing_steps
from .winnie3.search import WordMatch
from .winnie3.spacy_search import SpacyMatch
import yaml
from sqlalchemy import create_engine

import json


def create_conn():
    """Here we setup the database connection."""
    setup_dir = Path(".")
    credentials_path = setup_dir / "conf" / "local" / "credentials.yml"
    with open(credentials_path, "r") as credential_file:
        credentials = yaml.safe_load(credential_file)

    parameter_path = setup_dir / "conf" / "base" / "parameters.yml"
    with open(parameter_path, "r") as parameter_file:
        parameters = yaml.safe_load(parameter_file)

    db_host = parameters["DATABASE_PARAMS"]["db_host"]
    db_name = parameters["DATABASE_PARAMS"]["db_name"]
    db_user = credentials["dssg"]["username"]
    db_pass = credentials["dssg"]["password"]

    conn = create_engine(
        "mysql+pymysql://%s:%s@%s/%s" % (db_user, db_pass, db_host, db_name),
        encoding="latin1",
        echo=True,
    )

    return conn


def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    conn = create_conn()

    raw_text = pd.read_parquet("/datadrive/models/raw_text")
    train_winnie_settings = {}
    train_winnie_settings["preprocessing_steps"] = [
        "denoise_text",
        "remove_punctuation",
        "replace_numbers_with_words",
        "remove_punctuation",
        "remove_custom_stop_words",
        "replace_words",
    ]

    spacymatch = SpacyMatch()
    spacymatch.load_state()

    # a simple page that says hello
    @app.route("/", methods=["GET", "POST"])
    def hello():

        if request.method == "POST":
            print("form:", request.form)
            query = request.form["query"]  # this is my question_series
            extra_words = request.form["extra_words"]  # this is my extra_words

            # checkanswers = "checkanswers" in request.form
            # checkquestions = "checkquestions" in request.form

            # metric = request.form["metric"]
            # overlap_checked = request.form["metric"] == "overlap"
            # cosine_checked = request.form["metric"] == "cosine"
            # euclidean_checked = request.form["metric"] == "euclidean"

            # startyear = int(request.form["startyear"])
            was_queried = len(query) > 0
            tokens = []
        else:
            query = ""
            extra_words = ""
            was_queried = False
            tokens = []

        if was_queried:
            question_series = Series([query])
            # process query
            question_series_pp = run_preprocessing_steps(
                series=question_series,
                steps=train_winnie_settings["preprocessing_steps"],
            )
            if extra_words:
                extra_words_list = list(
                    filter(lambda s: s, re.split(",", extra_words))
                )  #  should work with comma separated bigrams
            else:
                extra_words_list = []
            ind, dist = spacymatch.search(
                question_series_pp, extra_words=extra_words_list
            )

            # results = (ind[0], dist[0])  # hardcoded for now
            answers = raw_text["answer"]

            # query_results = [
            #    (doc_id, score, answers.iloc[doc_id])
            #    for (doc_id, score) in results[:50]
            # ]

            results = []
            for j in range(50):
                results.append(
                    dict(
                        bg_class="success",
                        text=answers.iloc[ind[0, j]],
                        score=f"{dist[0,j]:4f}",
                        no=j,
                    )
                )

            count = len(results)
            current_page = 1
            page_results = results[:10]

        else:
            count = 49
            current_page = 5
            page_results = [
                dict(
                    text=f"something <span class='fw-bold'>{i}</span> something",
                    bg_class="success" if i % 2 == 0 else "danger",
                )
                for i in range(10)
            ]
        params = dict(
            was_queried=was_queried,
            query=query,
            extra_words=extra_words,
            start_idx=(current_page - 1) * 10 + 1,
            end_idx=min(current_page * 10, count),
            count=count,
            current_page=current_page,
            back_disabled="disabled" if current_page == 1 else "",
            forward_disabled="disabled" if current_page == (count + 9) // 10 else "",
            page_results=page_results,
        )
        return render_template("base.html", **params)

    @app.route("/api", methods=["GET", "POST"])
    def api():

        if request.method == "POST":
            print("form:", request.form)
            query = request.form["query"]
            checkanswers = "checkanswers" in request.form
            checkquestions = "checkquestions" in request.form

            metric = request.form["metric"]
            overlap_checked = request.form["metric"] == "overlap"
            cosine_checked = request.form["metric"] == "cosine"
            euclidean_checked = request.form["metric"] == "euclidean"

            startyear = int(request.form["startyear"])
            case = int(request.form["case"]) if request.form["case"] else None
            case_text = ""
            if case:
                cursor = conn.execute(
                    sqlalchemy.text(
                        "SELECT consultation_highlights FROM cases WHERE id = :caseid"
                    ),
                    {"caseid": case},
                )
                sqlquery_result = cursor.fetchone()
                if sqlquery_result:
                    case_text = sqlquery_result[0]
            was_queried = len(query) > 0 or case_text
            tokens = []
        else:
            query = ""
            was_queried = False
            checkanswers = True
            checkquestions = True
            startyear = 2010
            case = None
            case_text = ""
            tokens = []
            metric = "cosine"
            overlap_checked = False
            cosine_checked = True
            euclidean_checked = False

        if was_queried:
            if case_text:
                preprocessing_steps = [  # TODO: move these from here elsewhere
                    "denoise_text",
                    "remove_punctuation",
                    "replace_contractions",
                    "replace_numbers_with_words",
                    "remove_custom_stop_words",
                ]
                preprocessed_case = run_preprocessing_steps(
                    series=Series([case_text]), steps=preprocessing_steps
                )[0]
                tokens = nltk.word_tokenize(preprocessed_case)

            terms = filter(lambda s: s, re.split("\s|(?<!\d)[,.](?!\d)", query))
            nice_to_have, must_have, must_not_have = copy(tokens), [], []
            labels = (("A",) if checkanswers else (())) + (
                ("Q",) if checkquestions else (())
            )

            for term in terms:
                if term.startswith("+"):
                    must_have.append(term[1:].strip())
                elif term.startswith("-"):
                    must_not_have.append(term[1:].strip())
                else:
                    nice_to_have.append(term)

            query_results = word_match.search(
                nice_to_have=set(nice_to_have),
                must_have=set(must_have),
                must_not_have=set(must_not_have),
                labels=labels,
                time_filter=date(startyear, 1, 1),
                metric=metric,
            )

            results = []
            for j, (doc_id, score, words) in enumerate(query_results):
                text = raw_qa["qa_dict_raw"][doc_id]
                for word in words:  # TODO: this should be done differently
                    text = text.replace(
                        f" {word} ", f" <span class='fw-bold'>{word}</span> "
                    )
                if doc_id[0] == "Q" and ("A", doc_id[1]) in raw_qa["qa_dict_raw"]:
                    is_q = True
                    details = raw_qa["qa_dict_raw"][("A", doc_id[1])]
                else:
                    is_q = False
                    details = ""
                results.append(
                    dict(
                        bg_class="success" if doc_id[0] == "Q" else "danger",
                        text=text,
                        score=f"{score:4f}",
                        is_q=is_q,
                        details=details,
                        no=j,
                    )
                )
            return json.dumps(results)
        else:
            count = 49
            current_page = 5
            page_results = [
                dict(
                    text=f"something <span class='fw-bold'>{i}</span> something",
                    bg_class="success" if i % 2 == 0 else "danger",
                )
                for i in range(10)
            ]

        params = dict(
            was_queried=was_queried,
            query=query,
            checkanswers="checked" if checkanswers else "",
            checkquestions="checked" if checkquestions else "",
            start_idx=(current_page - 1) * 10 + 1,
            end_idx=min(current_page * 10, count),
            count=count,
            current_page=current_page,
            back_disabled="disabled" if current_page == 1 else "",
            forward_disabled="disabled" if current_page == (count + 9) // 10 else "",
            page_results=page_results,
            startyear=startyear,
            case=case,
            case_text=case_text,
            overlap_checked="checked" if overlap_checked else "",
            cosine_checked="checked" if cosine_checked else "",
            euclidean_checked="checked" if euclidean_checked else "",
        )
        return render_template("base.html", **params)

    return app
