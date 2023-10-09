import re


def filter_outliers(data, max_num_words=200):
    """ Filter outlier question-answer pairs from training data
    :param data: Training data 
    :param max_num_words: Threshold for number of words in outlier question-answer pairs
    :return: Filtered training data to only contain question-answer pairs with < max_num_words words.
    """

    data['num_words_ques'] = [len(re.findall(r'\w+', ques))
                              for ques in data['question']]
    data['num_words_ans'] = [len(re.findall(r'\w+', ans))
                             for ans in data['answer']]
    data = data.loc[(data['num_words_ques'] < max_num_words) &
                    (data['num_words_ans'] < max_num_words)]
    data.drop(['num_words_ques', 'num_words_ans'], axis=1)
    data.reset_index(drop=True, inplace=True)
    return data
