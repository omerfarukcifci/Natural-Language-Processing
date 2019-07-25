from homework1 import UnigramModel, BigramModel,TrigramModel
import re, random
import glob,errno

# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"
files = glob.glob('../data/*.txt')
testfiles = glob.glob('../testdata/*.txt')


def read_sentences_from_file(file_path):
    with open(file_path, "r") as f:
        f = [re.sub(r'[\.]', ' </s> <s>', x) for x in f]
        return [re.split ("\s+", line.rstrip('\n')) for line in f]

def read_test_file(file_path):
    with open(file_path, "r") as f:
        print("--------------------------------"+f.name+"--------------------------------")
        f = [re.sub(r'[\.]', ' </s> <s>', x) for x in f]
        return [re.split ("\s+", line.rstrip('\n')) for line in f]

def arrange_probabilities(model,modelNo):

    rangeStart = 0
    rangeEnd = 0
    probLine = dict()

    if modelNo == 1:
        for key, value in model.unigram_frequencies.items():
            rangeEnd += model.calculate_unigram_probability(key)
            probLine[(rangeStart, rangeEnd)] = key
            rangeStart = rangeEnd
    if modelNo == 2:
        for key, value in model.bigram_frequencies.items():
            rangeEnd += model.calculate_bigram_probabilty(key[0], key[1])
            probLine[(rangeStart, rangeEnd)] = key
            rangeStart = rangeEnd
    if modelNo == 3:
        for key, value in model.trigram_frequencies.items():
            rangeEnd += model.calculate_trigram_probabilty(key[0], key[1], key[2])
            probLine[(rangeStart, rangeEnd)] = key
            rangeStart = rangeEnd
    return probLine


def generate_essays(probLine, modelType):

    bigramText1 = ""
    bigramText2 = ""
    unigramText1 = ""
    unigramText2 = ""
    trigramText1 = ""
    trigramText2 = ""
    if modelType == 1:
        for i in range(30):
            randomNum = random.uniform(0, 1)
            for key, value in probLine.items():
                if key[0] < randomNum < key[1]:
                    if value != SENTENCE_START and value != SENTENCE_END:
                        unigramText1 = unigramText1 + "" + str(value) + " "
        for i in range(30):
            randomNum = random.uniform(0, 1)
            for key, value in probLine.items():
                if key[0] < randomNum < key[1]:
                    if value != SENTENCE_START and value != SENTENCE_END:
                        unigramText2 = unigramText2 + "" + str(value) + " "
        print("UNIGRAM ESSAY 1\t", unigramText1)
        print("UNIGRAM ESSAY 2\t", unigramText2)
    if modelType == 2:
        for i in range(15):
            randomNum = random.uniform(0, 1)
            for key, value in probLine.items():
                if key[0] < randomNum < key[1]:
                    if value[0] != SENTENCE_END and value[1] != SENTENCE_START:
                        bigramText1 = bigramText1 + "" + value[0] + " " + value[1] + " "
        for i in range(15):
            randomNum = random.uniform(0, 1)
            for key, value in probLine.items():
                if key[0] < randomNum < key[1]:
                    if value[0] != SENTENCE_END and value[1] != SENTENCE_START:
                        bigramText2 = bigramText2 + "" + value[0] + " " + value[1] + " "
        print("BIGRAM ESSAY 1\t", bigramText1)
        print("BIGRAM ESSAY 2\t", bigramText2)
    if modelType == 3:
        for i in range(10):
            randomNum = random.uniform(0, 1)
            for key, value in probLine.items():
                if key[0] < randomNum < key[1]:
                    if value[0] != SENTENCE_END and value[1] != SENTENCE_END and value[2] != SENTENCE_START:
                        trigramText1 = trigramText1 + "" + value[0] + " " + value[1] + " " + value[2] + " "
        for i in range(10):
            randomNum = random.uniform(0, 1)
            for key, value in probLine.items():
                if key[0] < randomNum < key[1]:
                    if value[0] != SENTENCE_END and value[1] != SENTENCE_END and value[2] != SENTENCE_START:
                        trigramText2 = trigramText2 + "" + value[0] + " " + value[1] + " " + value[2] + " "
        print("TRIGRAM ESSAY 1\t", trigramText1)
        print("TRIGRAM ESSAY 2\t", trigramText2)


if __name__ == '__main__':

    hamiltonData = []
    madisonData = []
    for name in files:
        try:
            veri=read_sentences_from_file(name)
            if veri[0][0] =='HAMILTON':
                hamiltonData.append(veri[1])
            if veri[0][0] =='MADISON':
                madisonData.append(veri[1])
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise

    madisonData = [[x.lower() for x in a] for a in madisonData]
    madisonData = [[re.sub(r'[^<>/\w\s]','',x) for x in a] for a in madisonData]
    hamiltonData = [[x.lower() for x in a] for a in hamiltonData]
    hamiltonData = [[re.sub(r'[^<>/\w\s]', '', x) for x in a] for a in hamiltonData]


    hamiltonUnigram = UnigramModel.UnigramLanguageModel(hamiltonData,smoothing=True)
    madisonUnigram = UnigramModel.UnigramLanguageModel(madisonData,smoothing=True)
    hamiltonBigram = BigramModel.BigramLanguageModel(hamiltonData, smoothing=True)
    madisonBigram = BigramModel.BigramLanguageModel(madisonData, smoothing=True)
    hamiltonTrigram = TrigramModel.TrigramLanguageModel(hamiltonData,smoothing=True)
    madisonTrigram = TrigramModel.TrigramLanguageModel(madisonData,smoothing=True)

    print("--------------------------------GENERATED ASSAYS OF HAMILTON--------------------------------")
    probabilityLine1 = arrange_probabilities(hamiltonUnigram,1)
    generate_essays(probabilityLine1,1)
    probabilityLine2 = arrange_probabilities(hamiltonBigram,2)
    generate_essays(probabilityLine2, 2)
    probabilityLine3 = arrange_probabilities(hamiltonTrigram, 3)
    generate_essays(probabilityLine3, 3)
    print()
    print("--------------------------------GENERATED ASSAYS OF MADISON--------------------------------")
    probabilityLine4 = arrange_probabilities(madisonUnigram, 1)
    generate_essays(probabilityLine4, 1)
    probabilityLine5 = arrange_probabilities(madisonBigram, 2)
    generate_essays(probabilityLine5, 2)
    probabilityLine6 = arrange_probabilities(madisonTrigram, 3)
    generate_essays(probabilityLine6, 3)
    print()


    for test in testfiles:
        try:
            testData = read_test_file(test)
            testData = [[x.lower() for x in a] for a in testData]
            testData = [[re.sub(r'[^<>/\w\s]', '', x) for x in a] for a in testData]
            hamiltonBigramResult = BigramModel.calculate_bigram_perplexity(hamiltonBigram, testData[1])
            madisonBigramResult = BigramModel.calculate_bigram_perplexity(madisonBigram, testData[1])
            hamiltonTrigramResult = TrigramModel.calculate_trigram_perplexity(hamiltonTrigram, testData[1])
            madisonTrigramResult = TrigramModel.calculate_trigram_perplexity(madisonTrigram, testData[1])
            if hamiltonBigramResult < madisonBigramResult:
                print("(BIGRAM) DETECTED ---> HAMILTON\t\t REAL ---> ",testData[0][0].upper())
                print("Hamilton perplexity: ", hamiltonBigramResult)
                print("Madison  perplexity: ", madisonBigramResult)
            else:
                print("(BIGRAM) DETECTED ---> MADISON\t\t REAL ---> ", testData[0][0].upper())
                print("Hamilton perplexity: ", hamiltonBigramResult)
                print("Madison  perplexity: ", madisonBigramResult)
            if hamiltonTrigramResult < madisonTrigramResult:
                print("(TRIGRAM) DETECTED ---> HAMILTON\t\t REAL ---> ",testData[0][0].upper())
                print("Hamilton perplexity: ", hamiltonTrigramResult)
                print("Madison  perplexity: ", madisonTrigramResult)
            else:
                print("(TRIGRAM) DETECTED ---> MADISON\t\t REAL ---> ", testData[0][0].upper())
                print("Hamilton perplexity: ", hamiltonTrigramResult)
                print("Madison  perplexity: ", madisonTrigramResult)
            print("---------------------------------------------------------------------------------")
            print()
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
