import logging
import numpy as np
import pandas as pd
import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_datasets(training_path, testing_path):
    training_df = pd.read_csv(training_path)
    testing_df = pd.read_csv(testing_path)
    x_train, x_test, y_train, y_test = training_df.text, testing_df.text, training_df.sentiment, testing_df.sentiment
    vectorizer = TfidfVectorizer()
    logging.info("Vectorizing dataset")
    x_train = vectorizer.fit_transform(x_train)
    pickle.dump(vectorizer, open('vectorizer.p', 'wb'))
    logging.info("x_train shape: {}".format(x_train.shape))
    x_test = vectorizer.transform(x_test)
    logging.info("x_test shape: {}".format(x_test.shape))
    return x_train, x_test, y_train, y_test


def read_dataset(path):
    dataset = pd.read_csv(path)
    x_train, x_test, y_train, y_test = train_test_split(dataset.text, dataset.sentiment, random_state=0, test_size=0.15)
    vectorizer = TfidfVectorizer()
    logging.info("Vectorizing dataset")
    x_train = vectorizer.fit_transform(x_train)
    logging.info("x_train shape: {}".format(x_train.shape))
    x_test = vectorizer.transform(x_test)
    logging.info("x_test shape: {}".format(x_test.shape))
    return x_train, x_test, y_train, y_test


def training(model, training_vectors, training_labels):
    model.fit(training_vectors, np.array(training_labels))
    pickle.dump(model, open('model.p', 'wb'))
    training_predictions = model.predict(training_vectors)
    logging.info('Training predicted classes: {}'.format(np.unique(training_predictions)))
    logging.info('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
    logging.info('Training F1 score: {}'.format(f1_score(training_labels, training_predictions, average='weighted')))
    # cv_scores = cross_val_score(model, training_vectors, training_labels, cv=10)
    # logging.info('Cross Validation (10k) accuracy: {}'.format(cv_scores.mean()))
    return model


def prediction(classifier, testing_vectors, testing_labels):
    testing_predictions = classifier.predict(testing_vectors)
    logging.info('Testing predicted classes: {}'.format(np.unique(testing_predictions)))
    logging.info('Testing accuracy: {}'.format(accuracy_score(testing_labels, testing_predictions)))
    logging.info('Testing F1 score: {}'.format(f1_score(testing_labels, testing_predictions, average='weighted')))
    logging.info('Testing confusion matrix (): {}'.format(confusion_matrix(testing_labels, testing_predictions,
                                                                           labels=['joy', 'anger', 'sadness', 'love', 'fear', 'thankfulness'])))


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = read_datasets('data/tweets_6labels1.csv', 'data/text_emotion.csv')
    # x_train, x_test, y_train, y_test = read_dataset('data/tweets1.csv', 'data/emotion1.csv')
    # models = [MultinomialNB(), LinearSVC(random_state=0), LogisticRegression(random_state=0)]
    models = [LinearSVC(random_state=0)]
    for mdl in models:
        logging.info(mdl)
        cls = training(mdl, x_train, y_train)
        prediction(cls, x_test, y_test)
