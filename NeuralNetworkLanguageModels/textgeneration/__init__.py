from homework4 import BigramModel
import numpy as np
import random
import json,string
from keras.utils import to_categorical
import dynet as dy
import nltk

dataPath = "../data/unim_poem.json"
count_word = 0
DATA_SIZE = 5000
EPOCH = 20


# Reading json from file basically
def read_json_file(file_path):
    with open(file_path) as f:
        return json.load(f)


# create and return vocabulary according to poems
def create_vocab(data):
    vocab = dict()
    for d in data[0:DATA_SIZE]:
        d["poem"] = " EOL ".join(d["poem"].split("\n"))
        nltk_tokens = nltk.word_tokenize(d["poem"])
        for word in nltk_tokens:
            vocab[word] = vocab.get(word, 0) + 1
    global count_word
    count_word = len(vocab)
    for key in list(vocab):
        if vocab.get(key) <= 1:
            del vocab[key]
            vocab["UNK"] = vocab.get("UNK", 0) + 1
    # print(vocab)
    # print(len(vocab))
    return vocab


# This function creates and arranges bigrams to use in model training part
def create_bigrams(data):
    bigrams = []
    for d in data[0:DATA_SIZE]:
        d["poem"] = " EOL ".join(d["poem"].split("\n"))
        nltk_tokens = nltk.word_tokenize(d["poem"])
        bigrams = bigrams + list(nltk.bigrams(nltk_tokens))
    for bi in bigrams:
        if bi[0] == "EOL":
            bigrams.remove(bi)
    # print("Bigram Size : ",len(bigrams))
    return bigrams


def create_bigrams_for_perplexity(data):
    for d in data[0:DATA_SIZE]:
        d["poem"] = " EOL ".join(d["poem"].split("\n"))
        nltk_tokens = nltk.word_tokenize(d["poem"])
        bigram_model = BigramModel.BigramLanguageModel(nltk_tokens, smoothing=True)
        return bigram_model


# This function creates indexed vocabulary from the normal vocabulary.
# We need indexed vocabulary to use one hot vectors
def create_indexed_vocabulary(vocab):
    # Indexed Vocabulary changes in each execution. This may be a problem!!! (SOLVED)
    i = 0
    indexedVocab = dict()
    for x in vocab.items():
        indexedVocab[i] = x
        i += 1
    # print(indexedVocab)
    # print("VOCAB SIZE2 ", len(indexedVocab))
    return indexedVocab

# Deprecated. Vector size error !
# def create_one_hot_vectors(indexedVocab):
#     one_hot_vectors = []
#     for x, y in indexedVocab.items():
#         one_hot_vectors.append(encode(x))
#     return one_hot_vectors

# Create one hot vectors from indexed vocabulary
def create_one_hot_vectors(indexedVocab):
    # indexedVocab = np.asarray(indexedVocab)
    index = []
    for x, y in list(indexedVocab.items()):
        index.append(x)
    index = np.asarray(index)
    one_hot_vectors = np.zeros((len(indexedVocab), len(indexedVocab)), dtype=np.uint8)
    one_hot_vectors[np.arange(len(indexedVocab)), index] = 1
    return one_hot_vectors


# Encode one hot vectors
def encode(x):
    encoded = to_categorical(x)
    return encoded


# Decode one hot vectors
def decode(x):
    return np.argmax(x)


# Get index of word from indexed vocabulary
def getIndexFromVocab(indexed_vocabulary,word):
    a = 0
    for i in indexed_vocab:
        if indexed_vocabulary[i][0] == word:
            return i


# Get word from indexed vocabulary according to index
def getWordFromIndexedVocab(indexed_vocabulary,index):
    return indexed_vocabulary[decode(one_hot_vecs[index])][0]


# Main part of pro
# gram. Create model and train model iteratively.
def training(bigrams,one_hot_vecs,indexed_vocab):

    model = dy.Model()
    vector_size = len(one_hot_vecs)
    input_size = vector_size
    hidden_size = int(vector_size/100)
    output_size = vector_size
    learning_rate = 0.1

    pW = model.add_parameters((hidden_size,output_size))
    pb = model.add_parameters((hidden_size))
    pU = model.add_parameters((input_size,hidden_size))
    pd = model.add_parameters((output_size))

    trainer = dy.SimpleSGDTrainer(model,learning_rate)

    for iteration in range(EPOCH):
        loss_iter = 0.0
        random.shuffle(bigrams)
        for bigram in bigrams:
            dy.renew_cg()
            bigram_first_index = getIndexFromVocab(indexed_vocab, bigram[0])
            bigram_second_index = getIndexFromVocab(indexed_vocab, bigram[1])
            if bigram_first_index != None and bigram_second_index != None:
                x = dy.inputVector(one_hot_vecs[bigram_first_index])
                # print("x : ", x.__sizeof__())
                y = pU * dy.tanh(pW * x + pb) + pd
                loss = dy.pickneglogsoftmax(y,bigram_second_index)
                # print(y,bigram_second_index,loss)
                loss_iter += loss.scalar_value()
                loss.forward()
                loss.backward()
                trainer.update()
        print("Iteration: ",iteration," Loss: ",loss_iter)

    model.save("my_train.model")


# Create poems using trained model
def create_poems(one_hot_vecs,bigram_model,indexed_vocab, line_number):
    model = dy.Model()
    dy.renew_cg()
    vector_size = len(one_hot_vecs)
    input_size = vector_size
    hidden_size = int(vector_size / 100)
    output_size = vector_size
    learning_rate = 0.1

    pW = model.add_parameters((hidden_size, output_size))
    pb = model.add_parameters((hidden_size))
    pU = model.add_parameters((input_size, hidden_size))
    pd = model.add_parameters((output_size))
    trainer = dy.SimpleSGDTrainer(model, learning_rate)

    a= model.populate("my_train.model")
    # print(a)
    for line in range(int(line_number)):
        rand = random.randrange(len(one_hot_vecs))
        predicted_word = ""
        poem = ""
        while predicted_word != "EOL":
            x = dy.inputVector(one_hot_vecs[rand])
            y = pU * dy.tanh(pW * x + pb) + pd
            rand = np.argmax(y.value())
            predicted_word = getWordFromIndexedVocab(indexed_vocab, np.argmax(y.value()))
            if predicted_word == "EOL":
                print()
                poem = poem + "\n"
            else:
                print(predicted_word,end = ' ')
                poem = poem + predicted_word + " "

        print("POEM PERPLEXITY", BigramModel.calculate_bigram_perplexity(bigram_model,poem))


if __name__ == '__main__':
    data = read_json_file(dataPath)
    vocab = create_vocab(data)
    bigrams = create_bigrams(data)
    indexed_vocab = create_indexed_vocabulary(vocab)
    one_hot_vecs = create_one_hot_vectors(indexed_vocab)
    bigram_model = create_bigrams_for_perplexity(data)

    print("TRAINING PROCESS RUNNING NOW ..\n")
    training(bigrams,one_hot_vecs,indexed_vocab)
    print("TRAINING PROCESS IS COMPLETED! \n")

    for i in range(5):
        line_of_poem = input("Enter line number of poem:")
        create_poems(one_hot_vecs,bigram_model,indexed_vocab, line_of_poem)