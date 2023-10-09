import nltk
import os
import logging
import os


"""Setting up logging"""
logging.basicConfig(filename='winnie.log', level=logging.INFO)

""" Downloading the nltk resources"""
logging.info('Setting up NLTK resouces...')

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

os.system("python -m spacy download en_core_web_sm")
os.system("pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder/releases/download/v0.4.5/en_use_md-0.4.5.tar.gz#en_use_md-0.4.5")

