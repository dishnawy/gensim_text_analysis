"""
    This class represents Floralytics NLP Engine
    ** note :this class is no stick with the en_wiki files only
"""
import logging
import os
import pickle
import re

from textblob import TextBlob

import gensim
import spacy
from gensim.summarization import keywords, summarize
from pyfasttext import FastText

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class NLP:
    
    def __init__(self):
        self.__version__ = '0.1'
        try:
            self.spacy_nlp = spacy.load('en')
        except:
            raise Exception("Can't find spacy corpus")

        try:
            # file name may change from machine to machine
            self.id2word = gensim.corpora.Dictionary.load_from_text(
                os.path.abspath('models/_wordids.txt.bz2'))
            self.mmCorpus = gensim.corpora.MmCorpus(
                os.path.abspath('models/_tfidf.mm'))
        except FileNotFoundError:
            raise Exception("Can't find a dataset!")

        try:
            self.lda = gensim.models.ldamodel.LdaModel.load(os.path.abspath('models/topic_modeling_lda_en_wiki_flora_december'))
        except FileNotFoundError:
            self.lda = None
            self.id2word = None
            self.mmCorpus = None
            logging.info("Can't find saved LDA model. Call train() function")
        try:
            self.lsi = gensim.models.lsimodel.LsiModel.load('topic_modeling_lsi_en_wiki')
        except:
            logging.info("Can't find saved LSI model. Call train() function")
        try:
            self.sentiment = FastText("//workspace/floralytics/NLP/fastText-0.1.0/amazonreviews/amazon_tuned.bin")
        except RuntimeError:
            logging.info("Can't find trained fasttext sentiment model")

        try:
            self.emotion_vectorizer = pickle.load(open('nlp_engine/emotion-analysis/vectorizer.p', 'rb'))
            self.emotion_model = pickle.load(open('nlp_engine/emotion-analysis/model.p', 'rb'))
        except RuntimeError:
            logging.info("Can't find trained emotion model or vectorizer")

    def topic_model_lda_training(self):
        try:
            self.lda = gensim.models.ldamodel.LdaModel(
                corpus=self.mmCorpus, id2word=self.id2word,
                num_topics=400, update_every=1, chunksize=10000, passes=1)
            logging.info("Successfully trained")
        except:
            raise Exception("couldn't  train model")

        logging.info("saving the model")
        try:
            self.lda.save(fname='topic_modeling_lda_en_wiki')
            logging.info("Successfully saved")
            return True
        except:
            raise Exception("couldn't  save model")
            
    def topic_model_lsi_training(self):
        try:
            self.lsi = gensim.models.lsimodel.LsiModel(corpus=self.mmCorpus, id2word=self.id2word, num_topics=400)
            logging.info("Successfully trained")
        except:
            raise Exception("couldn't  train model")

        logging.info("saving the model")
        try:
            self.lsi.save(fname='topic_modeling_lsi_en_wiki')
            logging.info("Successfully saved")
            return True
        except:
            raise Exception("couldn't save model")

    def topic_modeling_guess_lda(self, doc):
        """
        :param doc:
        :return: mapped list of list with topics mapped with
                 doc for predicted topics with accuracy
            doc should be list otherwise throw error
        """
        if not type(doc) == list:
            return False
        clean_text = self.topic_modeling_prepare_input(doc)
        if not clean_text or not len(clean_text) > 0:
            return False
        results = []
        for raw in clean_text:
            topic = self.lda[raw]
            row = []
            for tp in topic:
                row.append([self.lda.show_topic(topicid=tp[0]), str(tp[1])])
            results.append(row)
        return results

    def topic_modeling_guess_lsi(self, doc):
        """
        :param doc:
        :return: mapped list of list with topics mapped with
                 doc for predicted topics with accuracy
            doc should be list otherwise throw error
        """
        if not type(doc) == list:
            return False
        bow = self.topic_modeling_prepare_input(doc)
        if not bow or not len(bow) > 0:
            return False
        matrix = sorted(self.lsi[bow[0]], key=lambda x: abs(x[1]), reverse=True)
        
        result = {}
        for i in range(10):
            for j in self.lsi.show_topic(matrix[i][0]):
                if j[0] in result.keys():
                    result[j[0]] = result[j[0]] + 1
                else:
                    result[j[0]] = 1
        import operator
        result = [i[0] for i in sorted(result.items(), key=operator.itemgetter(1), reverse=True)[:5]]
        return result
    
    
    def topic_modeling_prepare_input(self, text):
        """
        Tokenize and clean list of text
        better to add NER here
        :param text: raw text input
        :return: list of  bag-of-words for clean words
        """
        # tokenize words
        preprocessed_docs = []
        for raw in text:
            doc = self.spacy_nlp(raw)
            lemmas = [
                token.lemma_ for token in doc
                if not token.is_stop and not token.is_punct]
            if not len(lemmas) > 0:
                continue
            preprocessed_docs.append(lemmas)
        if not len(preprocessed_docs) > 0:
            return False
        try:
            return [self.id2word.doc2bow(_doc) for _doc in preprocessed_docs]
        except:
            logging.info("couldn't convert preprocessed text to bag-of-word")
            return False

    def topic_modeling_update_model(self):
        raise Exception('Not implemented yet')

    def automatic_summarization_summarize(self, text, ratio=None):
        """
        Summarize a long english text
        :param text: raw text input
        :param ration [optional]: ratio of output to the input
        :return: str the summarized text
        """
        return summarize(text) if ratio is None else summarize(text, ratio)

    def automatic_summarization_keywords(self, text):
        ''' return list top keywords appears to be important in the text'''
        return keywords(text).split('\n')

    def fasttext_sentiment_predict(self, text):
        """
        Predict sentiment analysis for a given input using FastText supervised model
        :param text: input text
        :return: List of Positive & Negative tuples scores
        """
        text = text.lower().strip()
        # In order to get the same probabilities as the fastText binary
        # you have to add a newline (\n) at the end of input string.
        text += '\n'
        prediction = []
        predicted = self.sentiment.predict_proba_single(text, k=None, normalized=True)
        if predicted[0][0] == '1':
            # predicted negative sentiment
            negative = ('Negative', predicted[0][1])
            positive = ('Positive', predicted[1][1])
            prediction.append(negative)
            prediction.append(positive)
        else:
            # predicted positive sentiment
            negative = ('Negative', predicted[1][1])
            positive = ('Positive', predicted[0][1])
            prediction.append(positive)
            prediction.append(negative)
        return prediction

    def textblob_sentiment_predict(self, text):
        """
        Predict sentiment analysis for a given input using pre-trained TextBlob model
        :param text: input text
        :return: Sentiment polarity on a scale from -1 (very neg) to 1 (very pos)
        """
        text = text.lower().strip()
        text = TextBlob(text)
        return text.sentiment.polarity

    def emotion_predict(self, text):
        def preprocess(txt):
            return ' '.join(re.sub("(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", "", text.lower()).split())
        txt = preprocess(text)
        vectorized = self.emotion_vectorizer.transform([txt])
        return self.emotion_model.predict(vectorized)
