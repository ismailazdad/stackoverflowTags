import sys
from flask import Flask, jsonify, request, render_template, Config
import pandas as pd
import joblib
import itertools
import spacy
import en_core_web_sm
import datetime
from datetime import datetime
import os
import numpy
from service.preprocessing import Preprocessing
from service.w2v_features import w2vFeatures
from service.extractResults import ShowResultS



from gensim.models import Word2Vec

app = Flask(__name__)


model_path = "modeles/"
multilabel_binarizer = joblib.load(model_path + "multilabel_binarizer.pkl", 'r')

#modele et vecteurs TFIDF
tfidf_vectorizer = joblib.load(model_path + "tfidf_vectors.pkl", 'r')
model_logistic_tfidf = joblib.load(model_path + "model_logistic_tfidf.pkl", 'r')
model_sgdc_tfidf = joblib.load(model_path + "model_SGDC_tfidf.pkl", 'r')

#modele et vecteurs word2vec
word2vec_vectorizer = Word2Vec.load(model_path +"w2v_vertors.bin")
model_logistic_word2vec = joblib.load(model_path + "model_logistic_w2v.pkl", 'r')
model_sgdc_word2vec = joblib.load(model_path + "model_SGDC_w2v.pkl", 'r')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/hello')
def hello():
    return render_template('index.html', utc_dt=datetime.utcnow())

@app.route('/tagGenerators', methods=['POST'])
def tagGenerators():
    print('passage controller')
    result = {}
    question  = request.form.get("question")
    print(question)
    questiontext =  Preprocessing()
    cleaned_question =  questiontext.text_cleaner(question)
    print('result cleaned {}'.format(cleaned_question))
    extra_result_service =  ShowResultS()
    print('getTagTfIdfResult')
    tag_tfid = extra_result_service.getTagTfIdfResult(cleaned_question, tfidf_vectorizer, model_logistic_tfidf)
    tag_tfid_prob = extra_result_service.getTagTfIdfResultWithProba(cleaned_question, tfidf_vectorizer, model_logistic_tfidf)
    result['tfidf'] = tag_tfid
    result['tfidf_proba'] =  tag_tfid_prob['Pred_Tags_Probab']
    print('word2vec')
    tag_w2v = extra_result_service.getTagWord2vecResult(cleaned_question, word2vec_vectorizer, model_logistic_word2vec)
    tag_w2v_prob = extra_result_service.getTagWord2vecResultWithProba(cleaned_question, word2vec_vectorizer, model_logistic_word2vec)
    result['w2v'] = tag_w2v
    result['w2v_proba'] =  tag_w2v_prob['Pred_Tags_Probab']
    return result


if __name__ == "__main__":
    app.config['env'] = sys.argv[1]
    if app.config.get('env') == 'prod':
        app.run(host='178.170.47.69', port=5000)
    else:
        app.run(debug=True)
    # app.run(host='178.170.47.69', port=5000)
