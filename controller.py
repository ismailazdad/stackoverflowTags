import sys
from flask import Flask, jsonify, request, render_template, Config
import torch
from PIL import Image
import datetime
from datetime import datetime

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/hello')
def hello():
    return render_template('index.html', utc_dt=datetime.utcnow())



if __name__ == "__main__":
    app.config['env'] = sys.argv[1]
    if app.config.get('env') == 'prod':
        app.run(host='178.170.47.69', port=5000)
    else:
        app.run(debug=True)
    # app.run(host='178.170.47.69', port=5000)
