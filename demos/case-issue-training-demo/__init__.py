import json
import pickle
from random import randint, shuffle
from copy import copy
from flask import Flask, render_template, request
from .winnie3.case_issues_predict import predict_issues

TRAIN_WINNIE_SETTINGS = {
    "preprocessing_steps": [
        "denoise_text",
        "remove_punctuation",
        "replace_numbers_with_words",
        "remove_punctuation",
        "remove_custom_stop_words",
        "replace_words",
    ]
}


def create_app():
    # create and configure the app
    flask_app = Flask(__name__, instance_relative_config=True)

    preprocessing_steps = TRAIN_WINNIE_SETTINGS["preprocessing_steps"]
    with open("/datadrive/issues/ch_vectorizer_model", "rb") as f:
        vectorizer_model = pickle.load(f)
    with open("/datadrive/issues/ch_issue_predict_model", "rb") as f:
        issue_predict_model = pickle.load(f)
    with open("/datadrive/issues/ch_issue_predict_labels", "rb") as f:
        labels = pickle.load(f)

    def predict_labels(message):

        tags, tagsInfo = predict_issues(
            message,
            preprocessing_steps,
            vectorizer_model,
            issue_predict_model,
            labels,
        )
        result = []
        for ((_i, tag), prob), tagInfo in zip(tags, tagsInfo):
            score = int(round(prob * 100))
            confidence = (max(10, min(99, score)) // 10) * 10

            msg = copy(message)
            print(tagInfo)

            for word, importance in tagInfo:
                i = max(1, min(9, int(round(importance * 10))))
                msg = msg.replace(
                    word,
                    f'<span class="importance-{i * 10:d}" mx-1>{word}</span>',
                )

            result.append(
                {"text": tag, "score": score, "confidence": confidence, "tagInfo": msg}
            )

        result = sorted(result, key=lambda x: -x["score"])
        return result

    @flask_app.route("/", methods=["GET"])
    def index():
        new_message = randint(1, 10) % 2 == 0
        example_id = randint(1000, 9999)

        return render_template(
            "index.html",
            message="",
            header="New example" if new_message else f"Example {example_id}",
            is_new=new_message,
            case_issues=sorted(labels),
        )

    @flask_app.route("/predict_case_issues/", methods=["POST"])
    def predict_case_issues():
        full_message = request.form["inputText"]
        predicted_labels = predict_labels(full_message)
        return render_template("case_issues.html", predicted_labels=predicted_labels)

    return flask_app
