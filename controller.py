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
    question = "How can I remove a specific item from an array? I have an array of numbers and I'm using the method to add elements to it.\nIs there a simple way to remove a specific element from an array?\nI'm looking for the equivalent of something like:\n\nI have to use core JavaScript. Frameworks are not allowed"
    print('passage controller')
    t =  Preprocessing('')
    cleaned_question =  t.text_cleaner(question)
    print('result cleaned {}'.format(cleaned_question))
    return 'test'
# Cleaning function for new question

if __name__ == "__main__":
    app.config['env'] = sys.argv[1]
    if app.config.get('env') == 'prod':
        app.run(host='178.170.47.69', port=5000)
    else:
        app.run(debug=True)
    # app.run(host='178.170.47.69', port=5000)
