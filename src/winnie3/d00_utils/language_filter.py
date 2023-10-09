from langdetect import detect


def language_filter(data_series):
    detected_language = data_series.map(detect_language)
    data_series = data_series.loc[detected_language == "en"]
    return data_series


def detect_language(query):
    try:
        return detect(query)
    except:
        return "none"
