import pandas as pd
import spacy
import re


nlp = spacy.load("en_core_web_sm")
GREETING_REGEX = "(\\bdear\\b|\\bhi\\b|\\bhello\\b|\\bgood afternoon\\b|\\bgood morning\\b|\\bgood evening\\b)"


def personalise_recommendations(
    current_recommendations: pd.DataFrame,
    case_id: int,
    raw_cases: pd.DataFrame,
    intermediate_fb_messages: pd.DataFrame,
    intermediate_received_sms: pd.DataFrame,
):

    current_case_data = raw_cases.loc[raw_cases.id == case_id]

    beneficiary = detect_beneficiary_name(
        case_id=case_id,
        current_case_data=current_case_data,
        intermediate_fb_messages=intermediate_fb_messages,
        intermediate_received_sms=intermediate_received_sms,
    )
    current_recommendations["recommended_response"] = current_recommendations[
        "recommended_response"
    ].apply(remove_greeting)
    current_recommendations["recommended_response"] = current_recommendations[
        "recommended_response"
    ].apply(remove_mm)
    current_recommendations["recommended_response"] = current_recommendations[
        "recommended_response"
    ].apply(
        lambda text_value: replace_name(text_value=text_value, beneficiary=beneficiary)
    )
    recommendations = add_greeting(
        recommendations=current_recommendations, beneficiary=beneficiary
    )

    return recommendations


def replace_name(text_value: str, beneficiary: str):
    doc = nlp(text_value)
    terms = [clean_results(X.text) for X in doc.ents if X.label_ in {"PERSON"}]
    cleaned_text = text_value
    if len(terms) > 0:
        cleaned_text = cleaned_text.replace(terms[0], beneficiary)
        for (num, term) in enumerate(terms[1:]):
            cleaned_text = cleaned_text.replace(term, "Person_{}.".format(num + 1))
    return cleaned_text


def clean_results(text_value: str):
    cleaner_regex = re.compile(GREETING_REGEX, re.IGNORECASE)
    return cleaner_regex.sub("", text_value).strip()


def add_greeting(recommendations: pd.DataFrame, beneficiary: str):
    start_of_text = "(?:^)"
    optional_law = "(LAW\\s*)?"
    optional_name = "(\\s*\\w*\\s*\\w*)"
    punctuation = "(,|!|\\.|;)"
    regex = start_of_text + optional_law + GREETING_REGEX + optional_name + punctuation
    pat = re.compile(regex, re.IGNORECASE)

    recommendations["recommended_response"] = recommendations[
        "recommended_response"
    ].str.replace(pat, "")
    recommendations["recommended_response"] = recommendations[
        "recommended_response"
    ].str.strip()
    recommendations["recommended_response"] = recommendations[
        "recommended_response"
    ].str.capitalize()

    spacing = " " if len(beneficiary) > 0 else ""

    recommendations["recommended_response"] = (
        "Hi" + spacing + beneficiary + "! " + recommendations["recommended_response"]
    )

    return recommendations


def detect_name(text_value: str):
    doc = nlp(text_value)
    names = [X.text for X in doc.ents if X.label_ in {"PERSON"}]
    return names


def detect_beneficiary_name(
    case_id: int,
    current_case_data: pd.DataFrame,
    intermediate_fb_messages: pd.DataFrame,
    intermediate_received_sms: pd.DataFrame,
):
    """Detects the name of beneficiary.
    :param current_case_data: Pandas dataframe after fetching the case data
    :param intermediate_fb_messages: Pandas dataframe to get the name for facebook messages (if present)
    :param intermediate_received_sms: Pandas dataframe to get the name for messages (if present)
    :return: Beneficiary's name as a string
    """
    service_delivery = current_case_data["service_delivery"].iloc[0]

    if service_delivery == "facebook":
        beneficiary = intermediate_fb_messages.loc[
            intermediate_fb_messages.case_id == case_id, "from"
        ]
    elif service_delivery == "sms":
        beneficiary = intermediate_received_sms.loc[
            intermediate_received_sms.case_id == case_id, "sender_name"
        ]
    else:
        beneficiary = pd.Series([])

    if len(beneficiary) > 0:
        beneficiary = str(beneficiary.iloc[0])
    else:
        beneficiary = ""

    return beneficiary


def remove_mm(string: str):
    return string.replace("messagemerge", "")


def remove_greeting(string: str):
    words = re.findall(
        r"(^LAW|Hi|Hello|Dear|Good morning|Good afternoon|Good evening)(.*)(,|!|\\.|;)",
        string,
    )
    if len(words) == 0:
        pass
    else:
        for word in words[0]:
            string = string.replace(word, "")
        string = string.lstrip()
    return string
