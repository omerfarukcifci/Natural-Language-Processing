from homework2 import UnigramModel, BigramModel, Viterbi
import re,math,time
import pandas as pd
from tabulate import tabulate
import time
import numpy as np



def read_sentences_from_file(file_path):
    with open(file_path, "r") as f:
        return [re.split("\s+", line.rstrip('\n')) for line in f]


def create_initial_probs(data):
    initProbsData = []
    initProbsSplit = []
    for d in data:
        x = d[0].split("/")
        initProbsSplit.append(x[1])

    initProbsData.append(initProbsSplit)
    ModelInitProbs = UnigramModel.UnigramLanguageModel(initProbsData,smoothing=True)
    # print(ModelInitProbs.unigram_frequencies)
    initProbs = dict()

    for a in ModelInitProbs.unigram_frequencies:
        initProbs[a] = ModelInitProbs.calculate_unigram_probability(a)

    print("Initial : ",initProbs)
    return initProbs


def create_transition_probs(data):
    print(data)
    split = []
    for sentence in data:
        split2 = []
        for word in sentence:
            x = word.split("/")
            split2.append(x[1])
        split.append(split2)

    # print("splitted : ",split)
    transProbsModel = BigramModel.BigramLanguageModel(split,smoothing=True)
    # print(transProbsModel.bigram_frequencies)
    transProbs = dict()
    sum=0
    for key,val in transProbsModel.bigram_frequencies.items():
        for state in states:
            transProbs[key[0] + "|" + state] = transProbsModel.calculate_bigram_probabilty(key[0], state)
    for x ,y in transProbs.items():
        sum += float(y)
    print("Transition : ",transProbs)
    # print("sum: ",sum)
    return transProbs


def create_emission_probs(data):
    # print(data)
    split = []
    for sentence in data:
        split2 = []
        for word in sentence:
            paired = []
            x = word.split("/")
            paired.append(x[0])
            paired.append(x[1])
            split.append(paired)
        # split.append(split2)
    # print(split)
    emissionProbsModel = BigramModel.BigramLanguageModel(split, smoothing=True)
    # print(emissionProbsModel.bigram_frequencies)
    emissionProbs = dict()
    sum = 0
    for key, val in emissionProbsModel.bigram_frequencies.items():
        for state in states:
            emissionProbs[key[0] + "|" + state] = emissionProbsModel.calculate_bigram_probabilty(key[0], state)
    for x, y in emissionProbs.items():
        sum += float(y)
    print("Emission : ",emissionProbs)
    # print("sum: ", sum)
    return emissionProbs


def generate_sequence(states, sequence_length):
    all_sequences = []
    nodes = []

    depth = sequence_length

    def gen_seq_recur(states, nodes, depth):

        if depth == 0:
            # print nodes
            all_sequences.append(nodes)
        else:
            for state in states:
                temp_nodes = list(nodes)
                temp_nodes.append(state)
                gen_seq_recur(states, temp_nodes, depth - 1)

    gen_seq_recur(states, [], depth)

    return all_sequences


def score_sequences(sequences, initial_probs, transition_probs, emission_probs, obs):
    best_score = -1
    best_sequence = None

    sequence_scores = []

    for seq in sequences:

        total_score = 1
        total_score_breakdown = []
        first = True
        for i in range(len(seq)):
            state_score = 1

            # compute transitition probability score
            if first == True:
                state_score *= initial_probs[seq[i]]

                # reset first flag
                first = False
            else:
                state_score *= transition_probs[seq[i] + "|" + seq[i - 1]]

            # add to emission probability score
            try:
                state_score *= emission_probs[obs[i] + "|" + seq[i]]
            except:
                state_score *= 1
                # state_score *= emission_probs[obs[i] + "|" + seq[i]]

            # update the total score
            # print state_score
            total_score_breakdown.append(state_score)
            total_score *= state_score

        sequence_scores.append(total_score)

    return sequence_scores





def pretty_print_probs(distribs):
    rows = set()
    cols = set()
    for val in distribs.keys():
        temp = val.split("|")
        rows.add(temp[0])
        cols.add(temp[1])

    rows = list(rows)
    cols = list(cols)

    df = []
    for i in range(len(rows)):
        temp = []
        for j in range(len(cols)):
            temp.append(distribs[rows[i] + "|" + cols[j]])

        df.append(temp)

    I = pd.Index(rows, name="rows")
    C = pd.Index(cols, name="cols")
    df = pd.DataFrame(data=df, index=I, columns=C)

    print
    tabulate(df, headers='keys', tablefmt='psql')


def initializeSequences(_obs):
    # Generate list of sequences

    seqLen = len(_obs)

    seqs = generate_sequence(states, seqLen)

    # Score sequences
    seq_scores = score_sequences(seqs, initial_probs, transition_probs, emission_probs, obs)

    return (seqLen, seqs, seq_scores)


def arrangeTestData(data):

    testSentence = []
    for sentence in data:
        split3 = []
        for word in sentence:
            x = word.split("/")
            split3.append(x[0])
        testSentence.append(split3)
    # print(testSentence)
    return testSentence

def getRealTags(data):

    realTags = []
    for sentence in data:
        split2 = []
        for word in sentence:
            x = word.split("/")
            split2.append(x[1])
        realTags.append(split2)
    # print(realTag)
    return realTags


if __name__ == '__main__':
    data = read_sentences_from_file("./metu.txt")

    # print(data)
    initial_probs = create_initial_probs(data)
    states = []
    for key, val in initial_probs.items():
        states.append(key)
    transition_probs = create_transition_probs(data)
    emission_probs = create_emission_probs(data)
    # Bence/Pron çok/Adv haklılar/Verb ./Punc
    obs = ['Bence', 'çok', 'haklılar', '.']

    # Viterbi.viterbi(states,obs,initial_probs,transition_probs,emission_probs)


    testData = read_sentences_from_file("./test.txt")
    print(testData)
    testSentences = arrangeTestData(testData)
    ActualTags = getRealTags(testData)
    countCorrect = 0
    countAll = 0
    for i in range(0,len(testSentences)):
        myTagGuess = Viterbi.viterbi(states, testSentences[i], initial_probs, transition_probs, emission_probs)
        actualTag = ActualTags[i]
        if(myTagGuess != None and actualTag != None):
            for j in range(0,len(myTagGuess)):
                if (len(myTagGuess) == len(actualTag)):
                    countAll += 1
                    if (myTagGuess[j] == actualTag[j]):
                        countCorrect += 1

    print("ACCURACY: %",100*countCorrect/countAll)

