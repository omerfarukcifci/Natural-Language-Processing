from homework1 import BigramModel
import math

# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"


class TrigramLanguageModel():
    def __init__(self, sentences, smoothing=False):
        BigramModel.BigramLanguageModel.__init__(self, sentences, smoothing)
        self.trigram_frequencies = dict()
        self.unique_trigrams = set()
        for sentence in sentences:
            previous_word = None
            previous_word2 = None
            for word in sentence:
                if previous_word != None and previous_word2 != None :
                    self.trigram_frequencies[(previous_word2,previous_word, word)] = \
                        self.trigram_frequencies.get((previous_word2,previous_word, word),
                                                                                                 0) + 1
                    if previous_word != SENTENCE_START and word != SENTENCE_END:
                        self.unique_trigrams.add((previous_word, word))
                previous_word2 = previous_word
                previous_word = word


        self.unique__trigram_words = len(self.trigram_frequencies)

    def calculate_trigram_probabilty(self,previous_word2, previous_word, word):
        trigram_word_probability_numerator = self.trigram_frequencies.get((previous_word2,previous_word, word), 0)
        trigram_word_probability_denominator = self.bigram_frequencies.get((previous_word2, previous_word), 0)
        if self.smoothing:
            trigram_word_probability_numerator += 1
            trigram_word_probability_denominator += self.corpus_length + self.unique__trigram_words
        return 0.0 if trigram_word_probability_numerator == 0 or trigram_word_probability_denominator == 0 else float(
            trigram_word_probability_numerator) / float(trigram_word_probability_denominator)

    def calculate_trigram_sentence_probability(self, sentence):
        trigram_sentence_probability_log_sum = 0
        previous_word = None
        previous_word2 = None
        for word in sentence:
            if previous_word != None and previous_word2 != None :
                trigram_word_probability = self.calculate_trigram_probabilty(previous_word2,previous_word, word)
                trigram_sentence_probability_log_sum += math.log(trigram_word_probability, 2)
            previous_word2 = previous_word
            previous_word = word
        return trigram_sentence_probability_log_sum


def calculate_trigram_perplexity(model, sentences):
    trigram_sentence_probability_log_sum = model.calculate_trigram_sentence_probability(sentences)
    perplexity = math.pow(2, trigram_sentence_probability_log_sum / -len(sentences))
    # print("PERPELEXITY  :: ",perplexity)
    return perplexity