from flask import Flask, jsonify
from nlp_engine.nlp import NLP
import urllib.request
import pandas as pd
nlp = NLP()
app = Flask(__name__)


@app.route('/topic/<string:text>')
def topic(text):

    return pd.DataFrame(nlp.topic_modeling_guess_lsi([urllib.request.unquote(text)])).to_json()

@app.route('/summary/<string:text>')
def summary(text):
    return pd.DataFrame(nlp.automatic_summarization_summarize(urllib.request.unquote(text))).to_json()

@app.route('/summary/keywords/<string:text>')
def summary_keyword(text):
    return pd.DataFrame(nlp.automatic_summarization_keywords(urllib.request.unquote(text))).to_json()

@app.route('/sentiment/<string:text>')
def sentiment(text):
    return pd.DataFrame(nlp.textblob_sentiment_predict(text)).to_json()

@app.route('/emotion/<string:text>')
def emotion(text):
    return pd.DataFrame(nlp.emotion_predict(text)).to_json()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)