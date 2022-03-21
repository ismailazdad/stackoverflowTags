import sys
from flask import Flask, jsonify, request, render_template, Config
import pandas as pd
import joblib
import spacy
import en_core_web_sm

import datetime
from datetime import datetime
import os

# from .preprocessing import Preprocessing

# from test.preprocessing import Preprocessing

from service.preprocessing import Preprocessing




app = Flask(__name__)
print(os.getcwd())

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/hello')
def hello():
    return render_template('index.html', utc_dt=datetime.utcnow())

@app.route('/tagGenerators')
def tagGenerators():
    # question  = request.form.get("question")
    # print(question)
    question = 'java html css'
    print('passage controller')
    t =  Preprocessing('java html css')
    cleaned_question =  t.text_cleaner(question)
    return 'test'
# Cleaning function for new question

if __name__ == "__main__":
    app.config['env'] = sys.argv[1]
    if app.config.get('env') == 'prod':
        app.run(host='178.170.47.69', port=5000)
    else:
        app.run(debug=True)
    # app.run(host='178.170.47.69', port=5000)
