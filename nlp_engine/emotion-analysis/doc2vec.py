import logging
import gensim
import numpy as np
import pandas as pd

from gensim.models import doc2vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
LabeledSentence = gensim.models.doc2vec.LabeledSentence


def label_sentences(sentences, label_type):
    """Gensim's Doc2Vec implementation requires each document/paragraph to have a label"""
    labeled = []
    for idx, v in enumerate(sentences):
        label = '%s_%s' % (label_type, idx)
        labeled.append(LabeledSentence(v, [label]))
    return labeled


def get_vectors(d2v, corpus, size, label_type):
    """
    Get training vectors from doc2vec model
    :param d2v: trained doc2vec model
    :param corpus: text corpus (training, testing)
    :param size: vectors desired size
    :param label_type: training/testing
    :return: vectors of corpus
    """
    vectors = np.zeros((len(corpus), size))
    for idx in range(0, len(corpus)):
        index = idx
        if label_type == 'Test':
            index = idx + len(x_train)
        prefix = 'All_' + str(index)
        vectors[idx] = d2v.docvecs[prefix]
    return vectors


def train_doc2vec(corpus):
    logging.info("Building Doc2Vec model")
    model = doc2vec.Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
    model.build_vocab(corpus)
    return model


def read_dataset(path):
    dataset = pd.read_csv(path)
    x_train, x_test, y_train, y_test = train_test_split(dataset.text, dataset.sentiment, random_state=0, test_size=0.15)
    tweets = x_train.tolist() + x_test.tolist()
    x_train = label_sentences(x_train, 'Train')
    x_test = label_sentences(x_test, 'Test')
    all = label_sentences(tweets, 'All')
    return x_train, x_test, y_train, y_test, all


def train_classifier(d2v, training_vectors, training_labels):
    logging.info("Train Doc2Vec on training set")
    d2v.train(training_vectors, total_examples=d2v.corpus_count, epochs=d2v.iter)
    train_vectors = get_vectors(d2v, training_vectors, 100, 'Train')
    model = LogisticRegression()
    model.fit(train_vectors, np.array(training_labels))
    training_predictions = model.predict(train_vectors)
    logging.info('Training predicted classes: {}'.format(np.unique(training_predictions)))
    logging.info('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
    logging.info('Training F1 score: {}'.format(f1_score(training_labels, training_predictions, average='weighted')))
    cv_scores = cross_val_score(model, train_vectors, training_labels, cv=10)
    logging.info('Cross Validation (10k) accuracy: {}'.format(cv_scores.mean()))
    return model


def test_classifier(d2v, classifier, testing_vectors, testing_labels):
    logging.info("Train Doc2Vec on testing set")
    d2v.train(testing_vectors, total_examples=d2v.corpus_count, epochs=d2v.iter)
    test_vectors = get_vectors(d2v, testing_vectors, 100, 'Test')
    testing_predictions = classifier.predict(test_vectors)
    logging.info('Testing predicted classes: {}'.format(np.unique(testing_predictions)))
    logging.info('Testing accuracy: {}'.format(accuracy_score(testing_labels, testing_predictions)))
    logging.info('Testing F1 score: {}'.format(f1_score(testing_labels, testing_predictions, average='weighted')))


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, all = read_dataset('data/tweets.csv')
    doc2vec_model = train_doc2vec(all)
    cls = train_classifier(doc2vec_model, x_train, y_train)
    test_classifier(doc2vec_model, cls, x_test, y_test)
