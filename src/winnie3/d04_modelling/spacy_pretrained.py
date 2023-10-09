import spacy
import numpy as np
import pandas as pd
from winnie3.d04_modelling.feature_generator import FeatureGenerator


class SpacyVectorizer(FeatureGenerator):
    """spaCy trained pipeline"""

    def __init__(self, pretrained_model):
        FeatureGenerator.__init__(self)
        self.pretrained_model = pretrained_model

    def fit_model(self, save_path=None):
        """load the pretrained model"""
        self.model = spacy.load(self.pretrained_model)

        """save the model"""
        if save_path is not None:
            self.save_model(save_path)

    def generate_features(
        self, intermediate_series: pd.Series, saved_model_path=None, get_vectors=True
    ):
        """Generates features
        intermediate_series: series of preprocessed questions/answers
        """
        if saved_model_path is not None:
            self.model = spacy.load(self.pretrained_model)

        docs = intermediate_series.map(self.model)

        if get_vectors:
            return np.stack(docs.map(self.get_spacy_vector))
        else:
            return docs.tolist()

    def get_spacy_vector(self, x):
        return x.vector
