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


from gensim.models import Word2Vec



app = Flask(__name__)
print(os.getcwd())


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
    question  = request.form.get("question")
    print(question)
    # # question = "How can I remove a specific item from an array? I have an array of numbers and I'm using the method to add elements to it.\nIs there a simple way to remove a specific element from an array?\nI'm looking for the equivalent of something like:\n\nI have to use core JavaScript. Frameworks are not allowed"
    print('passage controller')
    t =  Preprocessing('')
    cleaned_question =  t.text_cleaner(question)
    print('result cleaned {}'.format(cleaned_question))


    #prediction with TDIDF
    #make vectorization and prediction
    X_tfidf = tfidf_vectorizer.transform([cleaned_question])
    predict = model_logistic_tfidf.predict(X_tfidf)
    predict_probas = model_logistic_tfidf.predict_proba(X_tfidf)
    # Inverse multilabel binarizer
    tags_predict_tfidf = multilabel_binarizer.inverse_transform(predict)
    tag_tfid = list(itertools.chain(*tags_predict_tfidf))
    print(tag_tfid)



    #prediction with word2vec
    #make vectorization and prediction
    X_w2v = word2vec_features([cleaned_question], word2vec_vectorizer)
    predict_w2v = model_logistic_word2vec.predict(X_w2v)
    tags_predict_w2v = multilabel_binarizer.inverse_transform(predict_w2v)
    print('result w2v')
    # print(tags_predict_w2v)
    tag_w2v = list(itertools.chain(*tags_predict_w2v))
    print(tag_w2v)

    resultList= list(set(tag_tfid) | set(tag_w2v))
    print('result joined')
    print(resultList)

    return 'end'

def get_vect(word, model):
    try:
        return model.wv[word]
    except KeyError:
        return numpy.zeros((model.vector_size,))

def sum_vectors(phrase, model):
    return sum(get_vect(w, model) for w in phrase)

def word2vec_features(X, model):
    feats = numpy.vstack([sum_vectors(p, model) for p in X])
    return feats

if __name__ == "__main__":
    app.config['env'] = sys.argv[1]
    if app.config.get('env') == 'prod':
        app.run(host='178.170.47.69', port=5000)
    else:
        app.run(debug=True)
    # app.run(host='178.170.47.69', port=5000)
