import re,os,codecs
import math
import numpy as np
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

testDataPath = "../data/word-test.v1.txt"
pretrainedDataPath = "../data/GoogleNews-vectors-negative300.bin"


def read_data_from_file(file_path):
    with open(file_path, "r") as f:
        return [re.split ("\s+", line.rstrip('\n')) for line in f]


def read_bin_data_from_file(file_path):
    with open(file_path, "rb") as binary_file:
        # Read the whole file at once
        data = binary_file.read(32)

        base64_data = codecs.encode(data, 'base64')
        print(base64_data)
        print(base64_data.decode('utf-8'))


def obtain_guess_vector(vectors,w1,w2,w3):
    try:
        v1 = vectors.get_vector(w1)
        v2 = vectors.get_vector(w2)
        v3 = vectors.get_vector(w3)
    except:
        v1 = np.ones(300)
        v2 = np.ones(300)
        v3 = np.ones(300)

    return v1 - v2 + v3


def cosine_similarity(vector1,vector2):
    dotProduct = vector1.dot(vector2)
    sum1 = 0
    sum2 = 0
    for v1i in vector1:
        sum1 +=math.pow(v1i,2)
    for v2i in vector2:
        sum2 +=math.pow(v2i,2)
    crossProduct = math.sqrt(sum1) * math.sqrt(sum2)

    return dotProduct / crossProduct


def find_most_similar_word(vectors,guess_vec):
    similarity_max = 0
    most_similar_word = ""
    for word in vectors.vocab:
        similarity = cosine_similarity(guess_vec, vectors.get_vector(word))
        if (similarity > similarity_max):
            similarity_max = similarity
            most_similar_word = word

    print("most similar word :", most_similar_word)
    print("similarity :", similarity_max)
    return most_similar_word


if __name__ == '__main__':
    vectors = KeyedVectors.load_word2vec_format(pretrainedDataPath, binary=True,limit=100000)
    # KeyedVectors.load_word2vec_format()
    # # result = vectors.most_similar(positive=['Germany', 'Paris'],negative=['France'],topn=1)
    # # print(result)grandson granddaughter husband wife
    # guess_vector = obtain_guess_vector(vectors,"he","she","mother")
    # find_most_similar_word(vectors,guess_vector)

    # Read test data and arranged like only quad word containing elements remained.
    testData = read_data_from_file(testDataPath)
    testData.remove(testData[0])

    correctEstimation = 0

    for quad in testData:
        if(quad[0]==":"):
            testData.remove(quad)
        else:
            estimatedVector = obtain_guess_vector(vectors,quad[0],quad[1],quad[2])
            estimatedWord = find_most_similar_word(vectors,estimatedVector)
            if(estimatedWord == quad[3]):
                correctEstimation+=1

    print("Accuracy: ",correctEstimation/len(testData))