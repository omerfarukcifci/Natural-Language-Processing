from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import utils
import csv
import pandas
import multiprocessing

import nltk

# File path
DataPath = "../data/tagged_plots_movielens.csv"
# Initializing the variables
train_documents = []
test_documents = []

# Associating the tags(labels) with numbers
tags_index = {'sci-fi': 1 , 'action': 2, 'comedy': 3, 'fantasy': 4, 'animation': 5, 'romance': 6}


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


def read_arrange_data():
    i = 0
    with open(DataPath, 'r') as csvfile:
        moviereader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in moviereader:
            if i == 0:
                i += 1
                continue
            i += 1
            if i <= 2000:

                train_documents.append(TaggedDocument(words=tokenize_text(row[2]), tags=[tags_index.get(row[3], 8)]))

            else:
                test_documents.append(TaggedDocument(words=tokenize_text(row[2]),
                                                     tags=[tags_index.get(row[3], 8)]))
    # print(train_documents[0])


def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors


if __name__ == '__main__':

    read_arrange_data()

    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=1, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=cores, alpha=0.025,
                         min_alpha=0.001)
    model_dbow.build_vocab([x for x in train_documents])

    train_documents = utils.shuffle(train_documents)
    model_dbow.train(train_documents, total_examples=len(train_documents), epochs=30)
    model_dbow.save('./plotsModel.d2v')

    y_train, X_train = vector_for_learning(model_dbow, train_documents)
    y_test, X_test = vector_for_learning(model_dbow, test_documents)

    regression = LogisticRegression(n_jobs=1, C=1e5)
    regression.fit(X_train, y_train)
    y_pred = regression.predict(X_test)

    print('Testing accuracy for movie plots %s ' % accuracy_score(y_test, y_pred))