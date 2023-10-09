# Winnie 3.0: Response Recommendation System for BarefootLaw

The aim of this project is to improve the first-line legal aid processes of [BarefootLaw](https://barefootlaw.org/).

## Overview

In Uganda, geographical and financial barriers limit people’s access to legal advice and guidance. 
95% of the 3,040 Ugandan lawyers are based in the capital city, while 96% of the population lives outside of Kampala.
The ratio of lawyers to population is of 1 for 13,000 inhabitants (by comparison, the ratio in the UK is of 1/429).
Moreover, 21.4% of the population living under the national poverty line.
Thus, while 90% of the population will experience a legal problem over a 4-year period, only 1% will have access to expert advice from a lawyer.

However, mobile phone technology is widespread in the country, and BarefootLaw (BFL) is a not-for-profit organization that leverages this to provide free legal guidance to Ugandans via social media (Facebook) and SMS. People have been quick to make use of BFL’s services and as a result, the number of requests has been growing every year. This has led to looking for ways to use technology, in particular Natural Language Processing (NLP), to aid lawyers in their everyday work. 

In this project, we developed a system that takes an incoming question, and provides a set of recommended responses that the lawyer could use to draft the response to a beneficiary. The system is called Winnie, and it was approached as an information retrieval system where the question is the query and the historical question-answer pairs are the documents to be retrieved. The system was based on the text data of question-answer pairs provided by BFL.

## Why Winnie?

The name came from one of BFL's early beneficiaries, Winnie Kobusingye.
With the guidance of BarefootLaw, she stood up for herself and used her legal rights to overcome a multitude of hardships, leaving a big mark on the core of BFL. You can learn more about her story [here](https://www.youtube.com/watch?v=biT0IjVsoA8&ab_channel=BarefootLaw).

## How does Winnie work?

Winnie embeds preprocessed text into high-dimensional vectors using the pretrained Google's [Universal Sentence Encoder (USE) ](https://tfhub.dev/google/universal-sentence-encoder/4) (via [SpaCy](https://github.com/MartinoMensio/spacy-universal-sentence-encoder)). 
For each incoming query, Winnie retrieves the closest matching previously-asked questions using the k-nearest neighbours algorithm with cosine similarity metric. 
The corresponding recommended answers are then provided to the lawyers via a custom web-interface (built by the amazing [Okalany Daniel](https://github.com/dokalanyi) \& co). 

## Kedro
> This project uses [Kedro](https://kedro.readthedocs.io/en/stable/). For more information head to 
the `/docs` section [here](./docs/KEDRO.md).

## Code Organization

The code is organized into 6 submodules.

- **d00_utils**: Utility functions used throughout the pipeline
- **d02_intermediate**: Pre-processing the raw data
    1. drop unneeded dataframe columns
    2. remove outreach (blast) messages
    3. remove empty rows
- **d03_primary**: Creating the question \& answer dataset to build the model 
    1. identify distinct conversations
    2. merge consecutive messages into threads
    3. identify individual speakers 
- **d04_modelling**: Create feature vectors from text
    1. drop unpopulated question-answer pairs
    2. filter our long messages
    3. preprocess text 
         - denoise text ( strip html tags, urls \& emails )
         - spelling correction
         - remove contractions
         - remove punctuation
         - replace numbers with words
         - remove (custom) stop words
    4. remove non-english questions
    5. embed preprocessed text
- **d05_reporting**: Model inference
    1. inference Winnie
        - preprocess query text + any extra terms*
         - embed preprocessed query
         - use k-nearest neighbors to identify k most similar questions
         - personalise recommendations (add/remove greeting + beneficiary name) 
- **d07_pipelines**: defining the pipeline nodes

\* in addition to beneficiary's query, Winnie can also be queried with additional terms provided by the lawyers (powered by spacy's [PhraseMatcher](https://spacy.io/api/phrasematcher)), which can be used to boost the scores of recommended answers containing them


<!---
Can be used with any system if you have data. 
Our innovation around technology continues to present an opportunity to extend access to justice to every corner of Uganda and Africa, and help people access services with dignity- despite their socio-economic situation.
using disruptive technology to promote access to justice
Legal guidance is instrumental in meeting the 16 Sustainable Development Goals, in Uganda and beyond.
-->

## Acknowledgements
-  [Dagstuhl Seminar 19082: AI for the Social Good](https://www.dagstuhl.de/en/program/calendar/semhp/?semnr=19082), where this project started
-  [Data Science for Social Good (DSSG)](https://www.datascienceforsocialgood.org/): DataFest 2019 Summer Fellowship students (Kasun Amarasinghe, Carlos Caro, Nupoor Gandhi, Raphaelle Roffo), their Technical Mentor Maren Eckhoff & Project Manager Samantha Short, who built and deployed [Winnie 1.0](https://github.com/dssg/barefoot-winnie-public)
- Imperial College London Department of Bioengineering BSc & MSc project students who contributed to Winnie 2.0 
- Open source software community - in particular the team @ [SpaCy](https://github.com/explosion/spaCy) & [Martino Mensio](https://github.com/MartinoMensio), who integrated Google's Universal Sentence Encoder directly within SpaCy

## Citations
Tomašev, N., Cornebise, J., Hutter, F., Mohamed, S., Picciariello, A., Connelly, B., ... & Clopath, C. (2020). [AI for social good: unlocking the opportunity for positive impact.](https://www.nature.com/articles/s41467-020-15871-z) Nature Communications, 11(1), 1-6.
