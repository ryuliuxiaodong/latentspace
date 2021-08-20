import numpy
from nltk.tokenize import word_tokenize
from gensim.models.keyedvectors import KeyedVectors
import nltk
import tensorflow as tf
import math
from nltk.metrics import *
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import ngrams
from nltk.translate.chrf_score import sentence_chrf
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import seaborn
import matplotlib.pyplot as plt


negation_set = ["no", "not", "aren't", "isn't", "n't", "nobody", "nowhere"]
entity_set = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP']
attribution_set = ['DT', 'PDT', 'JJ', 'JJR', 'JJS', 'PRP$', 'POS']
modification_set = ['EX', 'MD', 'RB', 'RBR', 'RBS', 'RP', 'TO']
action_set = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def cos_sim(ls1, ls2):
    if len(ls1) != len(ls2):
        return 0
    else:
        numerator = 0.0
        denominator = 0.0
        denominator1 = 0.0
        denominator2 = 0.0
        length = len(ls1)
        for i in range(length):
            numerator += (ls1[i] * ls2[i])
            denominator1 += (ls1[i] * ls1[i])
            denominator2 += (ls2[i] * ls2[i])
        denominator = math.sqrt(denominator1) * math.sqrt(denominator2)
        if denominator == 0:
            return 1
        else:
            fraction = numerator / denominator
            return fraction

def e_distance(ls1, ls2):
    if len(ls1) != len(ls2):
        return 0
    else:
        v1 = numpy.array(ls1)
        v2 = numpy.array(ls2)
        return numpy.linalg.norm(v1 - v2)

class SharedParameters:

    def __init__(self, fastText_file, paraphrase_latent_space_with_digitcount_file, non_paraphrase_latent_space_with_digitcount_file, paraphrase_latent_space_without_digitcount_file, non_paraphrase_latent_space_without_digitcount_file):

        self.fastText = KeyedVectors.load_word2vec_format(fastText_file, binary=False)

        matrix_file = open(paraphrase_latent_space_with_digitcount_file, 'r')
        matrix_table = matrix_file.readlines()
        matrix_file.close()
        matrix_file = open(non_paraphrase_latent_space_with_digitcount_file, 'r')
        matrix_table2 = matrix_file.readlines()
        matrix_file.close()
        matrix_list = []
        matrix_list2 = []
        for record in matrix_table:
            tmp = []
            record = record.rstrip("\n")
            all_values = record.split('*')
            for value in all_values:
                tmp.append(float(value))
            matrix_list.append(tmp)
        self.paraphrase_latent_space_with_digitcount = matrix_list
        for record in matrix_table2:
            tmp = []
            record = record.rstrip("\n")
            all_values = record.split('*')
            for value in all_values:
                tmp.append(float(value))
            matrix_list2.append(tmp)
        self.non_paraphrase_latent_space_with_digitcount = matrix_list2

        matrix_file = open(paraphrase_latent_space_without_digitcount_file, 'r')
        matrix_table = matrix_file.readlines()
        matrix_file.close()
        matrix_file = open(non_paraphrase_latent_space_without_digitcount_file, 'r')
        matrix_table2 = matrix_file.readlines()
        matrix_file.close()
        matrix_list = []
        matrix_list2 = []
        for record in matrix_table:
            tmp = []
            record = record.rstrip("\n")
            all_values = record.split('*')
            for value in all_values:
                tmp.append(float(value))
            matrix_list.append(tmp)
        self.paraphrase_latent_space_without_digitcount = matrix_list
        for record in matrix_table2:
            tmp = []
            record = record.rstrip("\n")
            all_values = record.split('*')
            for value in all_values:
                tmp.append(float(value))
            matrix_list2.append(tmp)
        self.non_paraphrase_latent_space_without_digitcount = matrix_list2


class DefaultSettings:

    def __init__(self, device, shared_parameters):
        self.shared_parameters = shared_parameters
        with tf.device(device):
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
        self.matrix_holder = tf.placeholder(tf.float32)
        self.input_holder = tf.placeholder(tf.float32)
        self.final_inputs_holder = tf.matmul(self.matrix_holder, self.input_holder)
        self.lemmatizer = WordNetLemmatizer()

    def __get_latent_semantics_paraphrase_space_PI_with_digitcount(self, sentence, diff, word_len, digit_count, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        tokens = word_tokenize(sentence)
        if len(tokens) == 0:
            return 0
        words = [word.lower() for word in tokens]
        pos_tags = nltk.pos_tag(words)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in pos_tags:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if entity_count <= 1:
            if digit_count <= 4:
                entity = 1.5 * entity_total
            else:
                entity = 0.5 * entity_total
        else:
            if digit_count <= 4:
                entity = 1.5 * (entity_total / entity_count)
            else:
                entity = 0.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 0.5 * (attribution_total + modification_total)
        else:
            attribution = 0.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 1.5 * action_total
        else:
            action = 1.5 * (action_total / action_count)
        if diff <= 5 or word_len >= 23:
            logic = numpy.array([1.0 for i in range(300)])
        else:
            logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if j_dist <= 0.6:
            jaccard = numpy.array([1.0 for i in range(300)])
        else:
            jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.paraphrase_latent_space_with_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)

    def __get_latent_semantics_nonparaphrase_space_PI_with_digitcount(self, sentence, diff, word_len, digit_count, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        tokens = word_tokenize(sentence)
        if len(tokens) == 0:
            return 0
        words = [word.lower() for word in tokens]
        pos_tags = nltk.pos_tag(words)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in pos_tags:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if entity_count <= 1:
            if digit_count <= 4:
                entity = 0.5 * entity_total
            else:
                entity = 1.5 * entity_total
        else:
            if digit_count <= 4:
                entity = 0.5 * (entity_total / entity_count)
            else:
                entity = 1.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 1.5 * (attribution_total + modification_total)
        else:
            attribution = 1.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 0.5 * action_total
        else:
            action = 0.5 * (action_total / action_count)
        if diff > 5 or word_len < 23:
            logic = numpy.array([1.0 for i in range(300)])
        else:
            logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if j_dist > 0.6:
            jaccard = numpy.array([1.0 for i in range(300)])
        else:
            jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.non_paraphrase_latent_space_with_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)


    def __get_latent_semantics_paraphrase_space_PI_without_digitcount(self, sentence, diff, word_len, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        tokens = word_tokenize(sentence)
        if len(tokens) == 0:
            return 0
        words = [word.lower() for word in tokens]
        pos_tags = nltk.pos_tag(words)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in pos_tags:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if entity_count <= 1:
            entity = 1.5 * entity_total
        else:
            entity = 1.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 0.5 * (attribution_total + modification_total)
        else:
            attribution = 0.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 1.5 * action_total
        else:
            action = 1.5 * (action_total / action_count)
        if diff <= 5 or word_len >= 23:
            logic = numpy.array([1.0 for i in range(300)])
        else:
            logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if j_dist <= 0.6:
            jaccard = numpy.array([1.0 for i in range(300)])
        else:
            jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.paraphrase_latent_space_without_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)

    def __get_latent_semantics_nonparaphrase_space_PI_without_digitcount(self, sentence, diff, word_len, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        tokens = word_tokenize(sentence)
        if len(tokens) == 0:
            return 0
        words = [word.lower() for word in tokens]
        pos_tags = nltk.pos_tag(words)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in pos_tags:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if entity_count <= 1:
            entity = 0.5 * entity_total
        else:
            entity = 0.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 1.5 * (attribution_total + modification_total)
        else:
            attribution = 1.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 0.5 * action_total
        else:
            action = 0.5 * (action_total / action_count)
        if diff > 5 or word_len < 23:
            logic = numpy.array([1.0 for i in range(300)])
        else:
            logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if j_dist > 0.6:
            jaccard = numpy.array([1.0 for i in range(300)])
        else:
            jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.non_paraphrase_latent_space_without_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)


    def get_features_for_PI_task(self, sentence1, sentence2, with_digit_count=True):
        print("the current input sentence pair is:")
        print(sentence1)
        print(sentence2)
        print("-------------------------------------------")
        tokens1 = word_tokenize(sentence1)
        tokens2 = word_tokenize(sentence2)
        diff = abs(len(tokens1) - len(tokens2))
        word_len1 = len(tokens1)
        word_len2 = len(tokens2)
        words1 = [word.lower() for word in tokens1]
        pos_tags1 = nltk.pos_tag(words1)
        words2 = [word.lower() for word in tokens2]
        pos_tags2 = nltk.pos_tag(words2)
        words11 = []
        words22 = []

        digit_count = 0
        digit_count1 = 0
        digit_count2 = 0
        for word, pos in pos_tags1:
            if pos == 'CD':
                digit_count += 1
                digit_count1 += 1
            elif pos in entity_set:
                words11.append(word)
            elif pos in action_set:
                words11.append(word)
        for word, pos in pos_tags2:
            if pos == 'CD':
                digit_count += 1
                digit_count2 += 1
            elif pos in entity_set:
                words22.append(word)
            elif pos in action_set:
                words22.append(word)
        digit_diff = abs(digit_count1 - digit_count2)

        wmdist = self.shared_parameters.fastText.wmdistance(words11, words22)
        if math.isnan(wmdist) or math.isinf(wmdist):
            wmdist = 0

        entity_count1 = 0
        entity_count2 = 0
        action_count1 = 0
        action_count2 = 0
        for word, pos in pos_tags1:
            if pos in entity_set:
                entity_count1 += 1
            elif pos in action_set:
                action_count1 += 1
        for word, pos in pos_tags2:
            if pos in entity_set:
                entity_count2 += 1
            elif pos in action_set:
                action_count2 += 1
        entity_diff = abs(entity_count1 - entity_count2)
        action_diff = abs(action_count1 - action_count2)

        j_dist = jaccard_distance(set(words1), set(words2))

        sentence1_unigram = ngrams(words1, 1)
        sentence2_unigram = ngrams(words2, 1)
        overlap = len(set(sentence1_unigram).intersection(set(sentence2_unigram)))
        unigram_precision = overlap / word_len1
        unigram_recall = overlap / word_len2

        sentence1_bigram = ngrams(words1, 2)
        sentence2_bigram = ngrams(words2, 2)
        overlap = len(set(sentence1_bigram).intersection(set(sentence2_bigram)))
        bigram_precision = overlap / word_len1
        bigram_recall = overlap / word_len2

        sentence1_trigram = ngrams(words1, 3)
        sentence2_trigram = ngrams(words2, 3)
        overlap = len(set(sentence1_trigram).intersection(set(sentence2_trigram)))
        trigram_precision = overlap / word_len1
        trigram_recall = overlap / word_len2

        bleu_score_p1 = sentence_bleu([words1], words2, weights=(0.5, 0, 0.5, 0))
        bleu_score_r1 = sentence_bleu([words2], words1, weights=(0.5, 0, 0.5, 0))
        bleu_score_p2 = sentence_bleu([words1], words2, weights=(0, 0.5, 0, 0.5))
        bleu_score_r2 = sentence_bleu([words2], words1, weights=(0, 0.5, 0, 0.5))
        chrf1 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=5)
        chrf2 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=5)
        chrf3 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=6)
        chrf4 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=6)
        chrf7 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=4)
        chrf8 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=4)
        chrf9 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=3)
        chrf10 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=3)
        chrf11 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=2)
        chrf12 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=2)
        chrf13 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=1)
        chrf14 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=1)

        if with_digit_count is True:
            latent_semantics1_p = self.__get_latent_semantics_paraphrase_space_PI_with_digitcount(sentence1, diff, word_len1, digit_count, j_dist)
            latent_semantics2_p = self.__get_latent_semantics_paraphrase_space_PI_with_digitcount(sentence2, diff, word_len2, digit_count, j_dist)
            latent_semantics1_np = self.__get_latent_semantics_nonparaphrase_space_PI_with_digitcount(sentence1, diff, word_len1, digit_count, j_dist)
            latent_semantics2_np = self.__get_latent_semantics_nonparaphrase_space_PI_with_digitcount(sentence2, diff, word_len2, digit_count, j_dist)
            tmp = cos_sim(latent_semantics1_p, latent_semantics2_p)
            tmp2 = cos_sim(latent_semantics1_np, latent_semantics2_np)
            tmp3 = e_distance(latent_semantics1_p, latent_semantics1_np)
            tmp4 = e_distance(latent_semantics2_p, latent_semantics2_np)
            latent_s = numpy.concatenate(((latent_semantics1_p + latent_semantics2_p), abs(latent_semantics1_p - latent_semantics2_p), (latent_semantics1_np + latent_semantics2_np), abs(latent_semantics1_np - latent_semantics2_np))).tolist()
            latent_s.append(edit_distance(sentence1, sentence2))
            latent_s.append(wmdist)
            latent_s.append(tmp)
            latent_s.append(tmp2)
            latent_s.append(tmp3)
            latent_s.append(tmp4)
            latent_s.append(digit_diff)
            latent_s.append(entity_diff)
            latent_s.append(action_diff)
            latent_s.append(diff)
            latent_s.append(unigram_precision)
            latent_s.append(unigram_recall)
            latent_s.append(bigram_precision)
            latent_s.append(bigram_recall)
            latent_s.append(trigram_precision)
            latent_s.append(trigram_recall)
            latent_s.append(bleu_score_p1)
            latent_s.append(bleu_score_r1)
            latent_s.append(bleu_score_p2)
            latent_s.append(bleu_score_r2)
            latent_s.append(chrf1)
            latent_s.append(chrf2)
            latent_s.append(chrf3)
            latent_s.append(chrf4)
            latent_s.append(chrf7)
            latent_s.append(chrf8)
            latent_s.append(chrf9)
            latent_s.append(chrf10)
            latent_s.append(chrf11)
            latent_s.append(chrf12)
            latent_s.append(chrf13)
            latent_s.append(chrf14)

            return latent_s
        else:
            latent_semantics1_p = self.__get_latent_semantics_paraphrase_space_PI_without_digitcount(sentence1, diff, word_len1, j_dist)
            latent_semantics2_p = self.__get_latent_semantics_paraphrase_space_PI_without_digitcount(sentence2, diff, word_len2, j_dist)
            latent_semantics1_np = self.__get_latent_semantics_nonparaphrase_space_PI_without_digitcount(sentence1, diff, word_len1, j_dist)
            latent_semantics2_np = self.__get_latent_semantics_nonparaphrase_space_PI_without_digitcount(sentence2, diff, word_len2, j_dist)
            tmp = cos_sim(latent_semantics1_p, latent_semantics2_p)
            tmp2 = cos_sim(latent_semantics1_np, latent_semantics2_np)
            tmp3 = e_distance(latent_semantics1_p, latent_semantics1_np)
            tmp4 = e_distance(latent_semantics2_p, latent_semantics2_np)
            latent_s = numpy.concatenate(((latent_semantics1_p + latent_semantics2_p), abs(latent_semantics1_p - latent_semantics2_p), (latent_semantics1_np + latent_semantics2_np), abs(latent_semantics1_np - latent_semantics2_np))).tolist()
            latent_s.append(edit_distance(sentence1, sentence2))
            latent_s.append(wmdist)
            latent_s.append(tmp)
            latent_s.append(tmp2)
            latent_s.append(tmp3)
            latent_s.append(tmp4)
            latent_s.append(digit_diff)
            latent_s.append(entity_diff)
            latent_s.append(action_diff)
            latent_s.append(diff)
            latent_s.append(unigram_precision)
            latent_s.append(unigram_recall)
            latent_s.append(bigram_precision)
            latent_s.append(bigram_recall)
            latent_s.append(trigram_precision)
            latent_s.append(trigram_recall)
            latent_s.append(bleu_score_p1)
            latent_s.append(bleu_score_r1)
            latent_s.append(bleu_score_p2)
            latent_s.append(bleu_score_r2)
            latent_s.append(chrf1)
            latent_s.append(chrf2)
            latent_s.append(chrf3)
            latent_s.append(chrf4)
            latent_s.append(chrf7)
            latent_s.append(chrf8)
            latent_s.append(chrf9)
            latent_s.append(chrf10)
            latent_s.append(chrf11)
            latent_s.append(chrf12)
            latent_s.append(chrf13)
            latent_s.append(chrf14)

            return latent_s

    def __get_latent_semantics_paraphrase_space_NLISTS_with_digitcount(self, sentence, diff, word_len, digit_count, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in sentence:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if entity_count <= 1:
            if digit_count <= 4:
                entity = 1.5 * entity_total
            else:
                entity = 0.5 * entity_total
        else:
            if digit_count <= 4:
                entity = 1.5 * (entity_total / entity_count)
            else:
                entity = 0.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 0.5 * (attribution_total + modification_total)
        else:
            attribution = 0.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 1.5 * action_total
        else:
            action = 1.5 * (action_total / action_count)
        if diff <= 5 or word_len >= 23:
            logic = numpy.array([1.0 for i in range(300)])
        else:
            logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if j_dist <= 0.6:
            jaccard = numpy.array([1.0 for i in range(300)])
        else:
            jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.paraphrase_latent_space_with_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)

    def __get_latent_semantics_nonparaphrase_space_NLISTS_with_digitcount(self, sentence, diff, word_len, digit_count, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in sentence:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if entity_count <= 1:
            if digit_count <= 4:
                entity = 0.5 * entity_total
            else:
                entity = 1.5 * entity_total
        else:
            if digit_count <= 4:
                entity = 0.5 * (entity_total / entity_count)
            else:
                entity = 1.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 1.5 * (attribution_total + modification_total)
        else:
            attribution = 1.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 0.5 * action_total
        else:
            action = 0.5 * (action_total / action_count)
        if diff > 5 or word_len < 23:
            logic = numpy.array([1.0 for i in range(300)])
        else:
            logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if j_dist > 0.6:
            jaccard = numpy.array([1.0 for i in range(300)])
        else:
            jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.non_paraphrase_latent_space_with_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)

    def __get_latent_semantics_paraphrase_space_NLISTS_without_digitcount(self, sentence, diff, word_len, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in sentence:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if entity_count <= 1:
            entity = 1.5 * entity_total
        else:
            entity = 1.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 0.5 * (attribution_total + modification_total)
        else:
            attribution = 0.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 1.5 * action_total
        else:
            action = 1.5 * (action_total / action_count)
        if diff <= 5 or word_len >= 23:
            logic = numpy.array([1.0 for i in range(300)])
        else:
            logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if j_dist <= 0.6:
            jaccard = numpy.array([1.0 for i in range(300)])
        else:
            jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.paraphrase_latent_space_without_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)

    def __get_latent_semantics_nonparaphrase_space_NLISTS_without_digitcount(self, sentence, diff, word_len, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in sentence:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if entity_count <= 1:
            entity = 0.5 * entity_total
        else:
            entity = 0.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 1.5 * (attribution_total + modification_total)
        else:
            attribution = 1.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 0.5 * action_total
        else:
            action = 0.5 * (action_total / action_count)
        if diff > 5 or word_len < 23:
            logic = numpy.array([1.0 for i in range(300)])
        else:
            logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if j_dist > 0.6:
            jaccard = numpy.array([1.0 for i in range(300)])
        else:
            jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.non_paraphrase_latent_space_without_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)

    def get_features_for_NLI_STS_tasks(self, sentence1, sentence2, with_digit_count=True):
        print("the current input sentence pair is:")
        print(sentence1)
        print(sentence2)
        print("-------------------------------------------")
        tokens1 = word_tokenize(sentence1)
        tokens2 = word_tokenize(sentence2)
        diff = abs(len(tokens1) - len(tokens2))
        word_len1 = len(tokens1)
        word_len2 = len(tokens2)
        words1 = [word.lower() for word in tokens1]
        pos_tags1 = nltk.pos_tag(words1)
        words2 = [word.lower() for word in tokens2]
        pos_tags2 = nltk.pos_tag(words2)
        words11 = []
        words22 = []

        digit_count = 0
        digit_count1 = 0
        digit_count2 = 0
        for word, pos in pos_tags1:
            if pos == 'CD':
                digit_count += 1
                digit_count1 += 1
            elif pos in entity_set:
                words11.append(word)
            elif pos in action_set:
                words11.append(word)
        for word, pos in pos_tags2:
            if pos == 'CD':
                digit_count += 1
                digit_count2 += 1
            elif pos in entity_set:
                words22.append(word)
            elif pos in action_set:
                words22.append(word)
        digit_diff = abs(digit_count1 - digit_count2)

        wmdist = self.shared_parameters.fastText.wmdistance(words11, words22)
        if math.isnan(wmdist) or math.isinf(wmdist):
            wmdist = 0

        entity_count1 = 0
        entity_count2 = 0
        action_count1 = 0
        action_count2 = 0
        for word, pos in pos_tags1:
            if pos in entity_set:
                entity_count1 += 1
            elif pos in action_set:
                action_count1 += 1
        for word, pos in pos_tags2:
            if pos in entity_set:
                entity_count2 += 1
            elif pos in action_set:
                action_count2 += 1
        entity_diff = abs(entity_count1 - entity_count2)
        action_diff = abs(action_count1 - action_count2)

        j_dist = jaccard_distance(set(words1), set(words2))

        sentence1_unigram = ngrams(words1, 1)
        sentence2_unigram = ngrams(words2, 1)
        overlap = len(set(sentence1_unigram).intersection(set(sentence2_unigram)))
        unigram_precision = overlap / word_len1
        unigram_recall = overlap / word_len2

        sentence1_bigram = ngrams(words1, 2)
        sentence2_bigram = ngrams(words2, 2)
        overlap = len(set(sentence1_bigram).intersection(set(sentence2_bigram)))
        bigram_precision = overlap / word_len1
        bigram_recall = overlap / word_len2

        sentence1_trigram = ngrams(words1, 3)
        sentence2_trigram = ngrams(words2, 3)
        overlap = len(set(sentence1_trigram).intersection(set(sentence2_trigram)))
        trigram_precision = overlap / word_len1
        trigram_recall = overlap / word_len2

        bleu_score_p1 = sentence_bleu([words1], words2, weights=(0.5, 0, 0.5, 0))
        bleu_score_r1 = sentence_bleu([words2], words1, weights=(0.5, 0, 0.5, 0))
        bleu_score_p2 = sentence_bleu([words1], words2, weights=(0, 0.5, 0, 0.5))
        bleu_score_r2 = sentence_bleu([words2], words1, weights=(0, 0.5, 0, 0.5))
        chrf1 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=5)
        chrf2 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=5)
        chrf3 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=6)
        chrf4 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=6)
        chrf7 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=4)
        chrf8 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=4)
        chrf9 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=3)
        chrf10 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=3)
        chrf11 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=2)
        chrf12 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=2)
        chrf13 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=1)
        chrf14 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=1)

        tl1 = []
        tl2 = []
        for word, pos in pos_tags1:
            if word in words2:
                continue
            else:
                tl1.append((word, pos))
        for word, pos in pos_tags2:
            if word in words1:
                continue
            else:
                tl2.append((word, pos))

        if with_digit_count is True:
            latent_semantics1_p = self.__get_latent_semantics_paraphrase_space_NLISTS_with_digitcount(tl1, diff, word_len1, digit_count, j_dist)
            latent_semantics2_p = self.__get_latent_semantics_paraphrase_space_NLISTS_with_digitcount(tl2, diff, word_len2, digit_count, j_dist)
            latent_semantics1_np = self.__get_latent_semantics_nonparaphrase_space_NLISTS_with_digitcount(tl1, diff, word_len1, digit_count, j_dist)
            latent_semantics2_np = self.__get_latent_semantics_nonparaphrase_space_NLISTS_with_digitcount(tl2, diff, word_len2, digit_count, j_dist)
            tmp = cos_sim(latent_semantics1_p, latent_semantics2_p)
            tmp2 = cos_sim(latent_semantics1_np, latent_semantics2_np)
            tmp3 = e_distance(latent_semantics1_p, latent_semantics1_np)
            tmp4 = e_distance(latent_semantics2_p, latent_semantics2_np)

            latent_s = numpy.concatenate(((latent_semantics1_p + latent_semantics2_p), abs(latent_semantics1_p - latent_semantics2_p), (latent_semantics1_np + latent_semantics2_np), abs(latent_semantics1_np - latent_semantics2_np))).tolist()
            latent_s.append(edit_distance(sentence1, sentence2))
            latent_s.append(wmdist)
            latent_s.append(tmp)
            latent_s.append(tmp2)
            latent_s.append(tmp3)
            latent_s.append(tmp4)
            latent_s.append(digit_diff)
            latent_s.append(entity_diff)
            latent_s.append(action_diff)
            latent_s.append(diff)
            latent_s.append(unigram_precision)
            latent_s.append(unigram_recall)
            latent_s.append(bigram_precision)
            latent_s.append(bigram_recall)
            latent_s.append(trigram_precision)
            latent_s.append(trigram_recall)
            latent_s.append(bleu_score_p1)
            latent_s.append(bleu_score_r1)
            latent_s.append(bleu_score_p2)
            latent_s.append(bleu_score_r2)
            latent_s.append(chrf1)
            latent_s.append(chrf2)
            latent_s.append(chrf3)
            latent_s.append(chrf4)
            latent_s.append(chrf7)
            latent_s.append(chrf8)
            latent_s.append(chrf9)
            latent_s.append(chrf10)
            latent_s.append(chrf11)
            latent_s.append(chrf12)
            latent_s.append(chrf13)
            latent_s.append(chrf14)
            n1 = 0
            n2 = 0
            for word in words1:
                if word in negation_set:
                    n1 += 1
            for word in words2:
                if word in negation_set:
                    n2 += 1
            latent_s.append(abs(n1 - n2))
            syn = 0
            ant = 0
            for word, pos in tl1:
                try:
                    word = self.lemmatizer.lemmatize(word)
                except:
                    continue
                synonyms = []
                antonyms = []
                for syn1 in wn.synsets(word):
                    for l in syn1.lemmas():
                        synonyms.append(l.name())
                        if l.antonyms():
                            antonyms.append(l.antonyms()[0].name())
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in synonyms:
                            syn += 1
                    except:
                        continue
                comb_name = word + ".n.01"
                try:
                    hyponyms = wn.synset(comb_name).hyponyms()
                    word_hyponyms = [lemma.name() for synset in hyponyms for lemma in synset.lemmas()]
                except:
                    print("hyponyms are not found for this word: " + word)
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in word_hyponyms:
                            syn += 1
                    except:
                        continue
                try:
                    hypernyms = wn.synset(comb_name).hypernyms()
                    word_hypernyms = [lemma.name() for synset in hypernyms for lemma in synset.lemmas()]
                except:
                    print("hypernyms are not found for this word: " + word)
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in word_hypernyms:
                            syn += 1
                    except:
                        continue
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in antonyms:
                            ant += 1
                    except:
                        continue
            latent_s.append(syn)
            latent_s.append(ant)
            latent_s.append(word_len1)
            latent_s.append(word_len2)
            latent_s.append(len(tl1))
            latent_s.append(len(tl2))

            return latent_s
        else:
            latent_semantics1_p = self.__get_latent_semantics_paraphrase_space_NLISTS_without_digitcount(tl1, diff, word_len1, j_dist)
            latent_semantics2_p = self.__get_latent_semantics_paraphrase_space_NLISTS_without_digitcount(tl2, diff, word_len2, j_dist)
            latent_semantics1_np = self.__get_latent_semantics_nonparaphrase_space_NLISTS_without_digitcount(tl1, diff, word_len1, j_dist)
            latent_semantics2_np = self.__get_latent_semantics_nonparaphrase_space_NLISTS_without_digitcount(tl2, diff, word_len2, j_dist)
            tmp = cos_sim(latent_semantics1_p, latent_semantics2_p)
            tmp2 = cos_sim(latent_semantics1_np, latent_semantics2_np)
            tmp3 = e_distance(latent_semantics1_p, latent_semantics1_np)
            tmp4 = e_distance(latent_semantics2_p, latent_semantics2_np)

            latent_s = numpy.concatenate(((latent_semantics1_p + latent_semantics2_p), abs(latent_semantics1_p - latent_semantics2_p), (latent_semantics1_np + latent_semantics2_np), abs(latent_semantics1_np - latent_semantics2_np))).tolist()
            latent_s.append(edit_distance(sentence1, sentence2))
            latent_s.append(wmdist)
            latent_s.append(tmp)
            latent_s.append(tmp2)
            latent_s.append(tmp3)
            latent_s.append(tmp4)
            latent_s.append(digit_diff)
            latent_s.append(entity_diff)
            latent_s.append(action_diff)
            latent_s.append(diff)
            latent_s.append(unigram_precision)
            latent_s.append(unigram_recall)
            latent_s.append(bigram_precision)
            latent_s.append(bigram_recall)
            latent_s.append(trigram_precision)
            latent_s.append(trigram_recall)
            latent_s.append(bleu_score_p1)
            latent_s.append(bleu_score_r1)
            latent_s.append(bleu_score_p2)
            latent_s.append(bleu_score_r2)
            latent_s.append(chrf1)
            latent_s.append(chrf2)
            latent_s.append(chrf3)
            latent_s.append(chrf4)
            latent_s.append(chrf7)
            latent_s.append(chrf8)
            latent_s.append(chrf9)
            latent_s.append(chrf10)
            latent_s.append(chrf11)
            latent_s.append(chrf12)
            latent_s.append(chrf13)
            latent_s.append(chrf14)
            n1 = 0
            n2 = 0
            for word in words1:
                if word in negation_set:
                    n1 += 1
            for word in words2:
                if word in negation_set:
                    n2 += 1
            latent_s.append(abs(n1 - n2))
            syn = 0
            ant = 0
            for word, pos in tl1:
                try:
                    word = self.lemmatizer.lemmatize(word)
                except:
                    continue
                synonyms = []
                antonyms = []
                for syn1 in wn.synsets(word):
                    for l in syn1.lemmas():
                        synonyms.append(l.name())
                        if l.antonyms():
                            antonyms.append(l.antonyms()[0].name())
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in synonyms:
                            syn += 1
                    except:
                        continue
                comb_name = word + ".n.01"
                try:
                    hyponyms = wn.synset(comb_name).hyponyms()
                    word_hyponyms = [lemma.name() for synset in hyponyms for lemma in synset.lemmas()]
                except:
                    print("hyponyms are not found for this word: " + word)
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in word_hyponyms:
                            syn += 1
                    except:
                        continue
                try:
                    hypernyms = wn.synset(comb_name).hypernyms()
                    word_hypernyms = [lemma.name() for synset in hypernyms for lemma in synset.lemmas()]
                except:
                    print("hypernyms are not found for this word: " + word)
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in word_hypernyms:
                            syn += 1
                    except:
                        continue
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in antonyms:
                            ant += 1
                    except:
                        continue
            latent_s.append(syn)
            latent_s.append(ant)
            latent_s.append(word_len1)
            latent_s.append(word_len2)
            latent_s.append(len(tl1))
            latent_s.append(len(tl2))

            return latent_s

    def get_paraphrase_latent_representations_for_sentence_pair(self, sentence1, sentence2, with_digit_count=True):
        print("the current input sentence pair is:")
        print(sentence1)
        print(sentence2)
        print("-------------------------------------------")
        tokens1 = word_tokenize(sentence1)
        tokens2 = word_tokenize(sentence2)
        diff = abs(len(tokens1) - len(tokens2))
        word_len1 = len(tokens1)
        word_len2 = len(tokens2)
        words1 = [word.lower() for word in tokens1]
        pos_tags1 = nltk.pos_tag(words1)
        words2 = [word.lower() for word in tokens2]
        pos_tags2 = nltk.pos_tag(words2)

        digit_count = 0
        for word, pos in pos_tags1:
            if pos == 'CD':
                digit_count += 1
        for word, pos in pos_tags2:
            if pos == 'CD':
                digit_count += 1

        j_dist = jaccard_distance(set(words1), set(words2))

        if with_digit_count is True:
            latent_semantics1_p = self.__get_latent_semantics_paraphrase_space_PI_with_digitcount(sentence1, diff, word_len1, digit_count, j_dist)
            latent_semantics2_p = self.__get_latent_semantics_paraphrase_space_PI_with_digitcount(sentence2, diff, word_len2, digit_count, j_dist)
            return (latent_semantics1_p, latent_semantics2_p)
        else:
            latent_semantics1_p = self.__get_latent_semantics_paraphrase_space_PI_without_digitcount(sentence1, diff, word_len1, j_dist)
            latent_semantics2_p = self.__get_latent_semantics_paraphrase_space_PI_without_digitcount(sentence2, diff, word_len2, j_dist)
            return (latent_semantics1_p, latent_semantics2_p)

    def get_nonparaphrase_latent_representations_for_sentence_pair(self, sentence1, sentence2, with_digit_count=True):
        print("the current input sentence pair is:")
        print(sentence1)
        print(sentence2)
        print("-------------------------------------------")
        tokens1 = word_tokenize(sentence1)
        tokens2 = word_tokenize(sentence2)
        diff = abs(len(tokens1) - len(tokens2))
        word_len1 = len(tokens1)
        word_len2 = len(tokens2)
        words1 = [word.lower() for word in tokens1]
        pos_tags1 = nltk.pos_tag(words1)
        words2 = [word.lower() for word in tokens2]
        pos_tags2 = nltk.pos_tag(words2)

        digit_count = 0
        for word, pos in pos_tags1:
            if pos == 'CD':
                digit_count += 1
        for word, pos in pos_tags2:
            if pos == 'CD':
                digit_count += 1

        j_dist = jaccard_distance(set(words1), set(words2))

        if with_digit_count is True:
            latent_semantics1_np = self.__get_latent_semantics_nonparaphrase_space_PI_with_digitcount(sentence1, diff, word_len1, digit_count, j_dist)
            latent_semantics2_np = self.__get_latent_semantics_nonparaphrase_space_PI_with_digitcount(sentence2, diff, word_len2, digit_count, j_dist)
            return (latent_semantics1_np, latent_semantics2_np)
        else:
            latent_semantics1_np = self.__get_latent_semantics_nonparaphrase_space_PI_without_digitcount(sentence1, diff, word_len1, j_dist)
            latent_semantics2_np = self.__get_latent_semantics_nonparaphrase_space_PI_without_digitcount(sentence2, diff, word_len2, j_dist)
            return (latent_semantics1_np, latent_semantics2_np)


class AdjustedSettings:

    def __init__(self, device, shared_parameters):
        self.shared_parameters = shared_parameters
        with tf.device(device):
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
        self.matrix_holder = tf.placeholder(tf.float32)
        self.input_holder = tf.placeholder(tf.float32)
        self.final_inputs_holder = tf.matmul(self.matrix_holder, self.input_holder)
        self.lemmatizer = WordNetLemmatizer()

        self.digitcount_threshold = 4
        self.digitcount_reverse = False
        self.senlendiff_threshold = 5
        self.senlendiff_reverse = False
        self.senlen_threshold = 23
        self.senlen_reverse = False
        self.jdist_threshold = 0.6
        self.jdist_reverse = False


    def adjust_settings(self, digitcount_threshold=4, digitcount_reverse=False, senlendiff_threshold=5, senlendiff_reverse=False, senlen_threshold=23, senlen_reverse=False, jdist_threshold=0.6, jdist_reverse=False):
        self.digitcount_threshold = digitcount_threshold
        self.digitcount_reverse = digitcount_reverse
        self.senlendiff_threshold = senlendiff_threshold
        self.senlendiff_reverse = senlendiff_reverse
        self.senlen_threshold = senlen_threshold
        self.senlen_reverse = senlen_reverse
        self.jdist_threshold = jdist_threshold
        self.jdist_reverse = jdist_reverse

    def __get_latent_semantics_paraphrase_space_PI_with_digitcount(self, sentence, diff, word_len, digit_count, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        tokens = word_tokenize(sentence)
        if len(tokens) == 0:
            return 0
        words = [word.lower() for word in tokens]
        pos_tags = nltk.pos_tag(words)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in pos_tags:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if self.digitcount_reverse is False:
            if entity_count <= 1:
                if digit_count <= self.digitcount_threshold:
                    entity = 1.5 * entity_total
                else:
                    entity = 0.5 * entity_total
            else:
                if digit_count <= self.digitcount_threshold:
                    entity = 1.5 * (entity_total / entity_count)
                else:
                    entity = 0.5 * (entity_total / entity_count)
        else:
            if entity_count <= 1:
                if digit_count <= self.digitcount_threshold:
                    entity = 0.5 * entity_total
                else:
                    entity = 1.5 * entity_total
            else:
                if digit_count <= self.digitcount_threshold:
                    entity = 0.5 * (entity_total / entity_count)
                else:
                    entity = 1.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 0.5 * (attribution_total + modification_total)
        else:
            attribution = 0.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 1.5 * action_total
        else:
            action = 1.5 * (action_total / action_count)
        if self.senlendiff_reverse is False and self.senlen_reverse is False:
            if diff <= self.senlendiff_threshold or word_len >= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is True and self.senlen_reverse is False:
            if diff >= self.senlendiff_threshold or word_len >= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is False and self.senlen_reverse is True:
            if diff <= self.senlendiff_threshold or word_len <= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if diff >= self.senlendiff_threshold or word_len <= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if self.jdist_reverse is False:
            if j_dist <= self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if j_dist >= self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.paraphrase_latent_space_with_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)

    def __get_latent_semantics_nonparaphrase_space_PI_with_digitcount(self, sentence, diff, word_len, digit_count, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        tokens = word_tokenize(sentence)
        if len(tokens) == 0:
            return 0
        words = [word.lower() for word in tokens]
        pos_tags = nltk.pos_tag(words)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in pos_tags:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if self.digitcount_reverse is False:
            if entity_count <= 1:
                if digit_count <= self.digitcount_threshold:
                    entity = 0.5 * entity_total
                else:
                    entity = 1.5 * entity_total
            else:
                if digit_count <= self.digitcount_threshold:
                    entity = 0.5 * (entity_total / entity_count)
                else:
                    entity = 1.5 * (entity_total / entity_count)
        else:
            if entity_count <= 1:
                if digit_count <= self.digitcount_threshold:
                    entity = 1.5 * entity_total
                else:
                    entity = 0.5 * entity_total
            else:
                if digit_count <= self.digitcount_threshold:
                    entity = 1.5 * (entity_total / entity_count)
                else:
                    entity = 0.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 1.5 * (attribution_total + modification_total)
        else:
            attribution = 1.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 0.5 * action_total
        else:
            action = 0.5 * (action_total / action_count)
        if self.senlendiff_reverse is False and self.senlen_reverse is False:
            if diff > self.senlendiff_threshold or word_len < self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is True and self.senlen_reverse is False:
            if diff < self.senlendiff_threshold or word_len < self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is False and self.senlen_reverse is True:
            if diff > self.senlendiff_threshold or word_len > self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if diff < self.senlendiff_threshold or word_len > self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if self.jdist_reverse is False:
            if j_dist > self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if j_dist < self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.non_paraphrase_latent_space_with_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)

    def __get_latent_semantics_paraphrase_space_PI_without_digitcount(self, sentence, diff, word_len, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        tokens = word_tokenize(sentence)
        if len(tokens) == 0:
            return 0
        words = [word.lower() for word in tokens]
        pos_tags = nltk.pos_tag(words)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in pos_tags:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if entity_count <= 1:
            entity = 1.5 * entity_total
        else:
            entity = 1.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 0.5 * (attribution_total + modification_total)
        else:
            attribution = 0.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 1.5 * action_total
        else:
            action = 1.5 * (action_total / action_count)
        if self.senlendiff_reverse is False and self.senlen_reverse is False:
            if diff <= self.senlendiff_threshold or word_len >= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is True and self.senlen_reverse is False:
            if diff >= self.senlendiff_threshold or word_len >= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is False and self.senlen_reverse is True:
            if diff <= self.senlendiff_threshold or word_len <= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if diff >= self.senlendiff_threshold or word_len <= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if self.jdist_reverse is False:
            if j_dist <= self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if j_dist >= self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.paraphrase_latent_space_without_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)

    def __get_latent_semantics_nonparaphrase_space_PI_without_digitcount(self, sentence, diff, word_len, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        tokens = word_tokenize(sentence)
        if len(tokens) == 0:
            return 0
        words = [word.lower() for word in tokens]
        pos_tags = nltk.pos_tag(words)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in pos_tags:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if entity_count <= 1:
            entity = 0.5 * entity_total
        else:
            entity = 0.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 1.5 * (attribution_total + modification_total)
        else:
            attribution = 1.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 0.5 * action_total
        else:
            action = 0.5 * (action_total / action_count)
        if self.senlendiff_reverse is False and self.senlen_reverse is False:
            if diff > self.senlendiff_threshold or word_len < self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is True and self.senlen_reverse is False:
            if diff < self.senlendiff_threshold or word_len < self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is False and self.senlen_reverse is True:
            if diff > self.senlendiff_threshold or word_len > self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if diff < self.senlendiff_threshold or word_len > self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if self.jdist_reverse is False:
            if j_dist > self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if j_dist < self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.non_paraphrase_latent_space_without_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)

    def get_features_for_PI_task(self, sentence1, sentence2, with_digit_count=True):
        print("the current input sentence pair is:")
        print(sentence1)
        print(sentence2)
        print("-------------------------------------------")
        tokens1 = word_tokenize(sentence1)
        tokens2 = word_tokenize(sentence2)
        diff = abs(len(tokens1) - len(tokens2))
        word_len1 = len(tokens1)
        word_len2 = len(tokens2)
        words1 = [word.lower() for word in tokens1]
        pos_tags1 = nltk.pos_tag(words1)
        words2 = [word.lower() for word in tokens2]
        pos_tags2 = nltk.pos_tag(words2)
        words11 = []
        words22 = []

        digit_count = 0
        digit_count1 = 0
        digit_count2 = 0
        for word, pos in pos_tags1:
            if pos == 'CD':
                digit_count += 1
                digit_count1 += 1
            elif pos in entity_set:
                words11.append(word)
            elif pos in action_set:
                words11.append(word)
        for word, pos in pos_tags2:
            if pos == 'CD':
                digit_count += 1
                digit_count2 += 1
            elif pos in entity_set:
                words22.append(word)
            elif pos in action_set:
                words22.append(word)
        digit_diff = abs(digit_count1 - digit_count2)

        wmdist = self.shared_parameters.fastText.wmdistance(words11, words22)
        if math.isnan(wmdist) or math.isinf(wmdist):
            wmdist = 0

        entity_count1 = 0
        entity_count2 = 0
        action_count1 = 0
        action_count2 = 0
        for word, pos in pos_tags1:
            if pos in entity_set:
                entity_count1 += 1
            elif pos in action_set:
                action_count1 += 1
        for word, pos in pos_tags2:
            if pos in entity_set:
                entity_count2 += 1
            elif pos in action_set:
                action_count2 += 1
        entity_diff = abs(entity_count1 - entity_count2)
        action_diff = abs(action_count1 - action_count2)

        j_dist = jaccard_distance(set(words1), set(words2))

        sentence1_unigram = ngrams(words1, 1)
        sentence2_unigram = ngrams(words2, 1)
        overlap = len(set(sentence1_unigram).intersection(set(sentence2_unigram)))
        unigram_precision = overlap / word_len1
        unigram_recall = overlap / word_len2

        sentence1_bigram = ngrams(words1, 2)
        sentence2_bigram = ngrams(words2, 2)
        overlap = len(set(sentence1_bigram).intersection(set(sentence2_bigram)))
        bigram_precision = overlap / word_len1
        bigram_recall = overlap / word_len2

        sentence1_trigram = ngrams(words1, 3)
        sentence2_trigram = ngrams(words2, 3)
        overlap = len(set(sentence1_trigram).intersection(set(sentence2_trigram)))
        trigram_precision = overlap / word_len1
        trigram_recall = overlap / word_len2

        bleu_score_p1 = sentence_bleu([words1], words2, weights=(0.5, 0, 0.5, 0))
        bleu_score_r1 = sentence_bleu([words2], words1, weights=(0.5, 0, 0.5, 0))
        bleu_score_p2 = sentence_bleu([words1], words2, weights=(0, 0.5, 0, 0.5))
        bleu_score_r2 = sentence_bleu([words2], words1, weights=(0, 0.5, 0, 0.5))
        chrf1 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=5)
        chrf2 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=5)
        chrf3 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=6)
        chrf4 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=6)
        chrf7 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=4)
        chrf8 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=4)
        chrf9 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=3)
        chrf10 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=3)
        chrf11 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=2)
        chrf12 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=2)
        chrf13 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=1)
        chrf14 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=1)

        if with_digit_count is True:
            latent_semantics1_p = self.__get_latent_semantics_paraphrase_space_PI_with_digitcount(sentence1, diff, word_len1, digit_count, j_dist)
            latent_semantics2_p = self.__get_latent_semantics_paraphrase_space_PI_with_digitcount(sentence2, diff, word_len2, digit_count, j_dist)
            latent_semantics1_np = self.__get_latent_semantics_nonparaphrase_space_PI_with_digitcount(sentence1, diff, word_len1, digit_count, j_dist)
            latent_semantics2_np = self.__get_latent_semantics_nonparaphrase_space_PI_with_digitcount(sentence2, diff, word_len2, digit_count, j_dist)
            tmp = cos_sim(latent_semantics1_p, latent_semantics2_p)
            tmp2 = cos_sim(latent_semantics1_np, latent_semantics2_np)
            tmp3 = e_distance(latent_semantics1_p, latent_semantics1_np)
            tmp4 = e_distance(latent_semantics2_p, latent_semantics2_np)
            latent_s = numpy.concatenate(((latent_semantics1_p + latent_semantics2_p), abs(latent_semantics1_p - latent_semantics2_p), (latent_semantics1_np + latent_semantics2_np), abs(latent_semantics1_np - latent_semantics2_np))).tolist()
            latent_s.append(edit_distance(sentence1, sentence2))
            latent_s.append(wmdist)
            latent_s.append(tmp)
            latent_s.append(tmp2)
            latent_s.append(tmp3)
            latent_s.append(tmp4)
            latent_s.append(digit_diff)
            latent_s.append(entity_diff)
            latent_s.append(action_diff)
            latent_s.append(diff)
            latent_s.append(unigram_precision)
            latent_s.append(unigram_recall)
            latent_s.append(bigram_precision)
            latent_s.append(bigram_recall)
            latent_s.append(trigram_precision)
            latent_s.append(trigram_recall)
            latent_s.append(bleu_score_p1)
            latent_s.append(bleu_score_r1)
            latent_s.append(bleu_score_p2)
            latent_s.append(bleu_score_r2)
            latent_s.append(chrf1)
            latent_s.append(chrf2)
            latent_s.append(chrf3)
            latent_s.append(chrf4)
            latent_s.append(chrf7)
            latent_s.append(chrf8)
            latent_s.append(chrf9)
            latent_s.append(chrf10)
            latent_s.append(chrf11)
            latent_s.append(chrf12)
            latent_s.append(chrf13)
            latent_s.append(chrf14)

            return latent_s
        else:
            latent_semantics1_p = self.__get_latent_semantics_paraphrase_space_PI_without_digitcount(sentence1, diff, word_len1, j_dist)
            latent_semantics2_p = self.__get_latent_semantics_paraphrase_space_PI_without_digitcount(sentence2, diff, word_len2, j_dist)
            latent_semantics1_np = self.__get_latent_semantics_nonparaphrase_space_PI_without_digitcount(sentence1, diff, word_len1, j_dist)
            latent_semantics2_np = self.__get_latent_semantics_nonparaphrase_space_PI_without_digitcount(sentence2, diff, word_len2, j_dist)
            tmp = cos_sim(latent_semantics1_p, latent_semantics2_p)
            tmp2 = cos_sim(latent_semantics1_np, latent_semantics2_np)
            tmp3 = e_distance(latent_semantics1_p, latent_semantics1_np)
            tmp4 = e_distance(latent_semantics2_p, latent_semantics2_np)
            latent_s = numpy.concatenate(((latent_semantics1_p + latent_semantics2_p), abs(latent_semantics1_p - latent_semantics2_p), (latent_semantics1_np + latent_semantics2_np), abs(latent_semantics1_np - latent_semantics2_np))).tolist()
            latent_s.append(edit_distance(sentence1, sentence2))
            latent_s.append(wmdist)
            latent_s.append(tmp)
            latent_s.append(tmp2)
            latent_s.append(tmp3)
            latent_s.append(tmp4)
            latent_s.append(digit_diff)
            latent_s.append(entity_diff)
            latent_s.append(action_diff)
            latent_s.append(diff)
            latent_s.append(unigram_precision)
            latent_s.append(unigram_recall)
            latent_s.append(bigram_precision)
            latent_s.append(bigram_recall)
            latent_s.append(trigram_precision)
            latent_s.append(trigram_recall)
            latent_s.append(bleu_score_p1)
            latent_s.append(bleu_score_r1)
            latent_s.append(bleu_score_p2)
            latent_s.append(bleu_score_r2)
            latent_s.append(chrf1)
            latent_s.append(chrf2)
            latent_s.append(chrf3)
            latent_s.append(chrf4)
            latent_s.append(chrf7)
            latent_s.append(chrf8)
            latent_s.append(chrf9)
            latent_s.append(chrf10)
            latent_s.append(chrf11)
            latent_s.append(chrf12)
            latent_s.append(chrf13)
            latent_s.append(chrf14)

            return latent_s

    def __get_latent_semantics_paraphrase_space_NLISTS_with_digitcount(self, sentence, diff, word_len, digit_count, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in sentence:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if self.digitcount_reverse is False:
            if entity_count <= 1:
                if digit_count <= self.digitcount_threshold:
                    entity = 1.5 * entity_total
                else:
                    entity = 0.5 * entity_total
            else:
                if digit_count <= self.digitcount_threshold:
                    entity = 1.5 * (entity_total / entity_count)
                else:
                    entity = 0.5 * (entity_total / entity_count)
        else:
            if entity_count <= 1:
                if digit_count <= self.digitcount_threshold:
                    entity = 0.5 * entity_total
                else:
                    entity = 1.5 * entity_total
            else:
                if digit_count <= self.digitcount_threshold:
                    entity = 0.5 * (entity_total / entity_count)
                else:
                    entity = 1.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 0.5 * (attribution_total + modification_total)
        else:
            attribution = 0.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 1.5 * action_total
        else:
            action = 1.5 * (action_total / action_count)
        if self.senlendiff_reverse is False and self.senlen_reverse is False:
            if diff <= self.senlendiff_threshold or word_len >= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is True and self.senlen_reverse is False:
            if diff >= self.senlendiff_threshold or word_len >= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is False and self.senlen_reverse is True:
            if diff <= self.senlendiff_threshold or word_len <= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if diff >= self.senlendiff_threshold or word_len <= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if self.jdist_reverse is False:
            if j_dist <= self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if j_dist >= self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.paraphrase_latent_space_with_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)

    def __get_latent_semantics_nonparaphrase_space_NLISTS_with_digitcount(self, sentence, diff, word_len, digit_count, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in sentence:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if self.digitcount_reverse is False:
            if entity_count <= 1:
                if digit_count <= self.digitcount_threshold:
                    entity = 0.5 * entity_total
                else:
                    entity = 1.5 * entity_total
            else:
                if digit_count <= self.digitcount_threshold:
                    entity = 0.5 * (entity_total / entity_count)
                else:
                    entity = 1.5 * (entity_total / entity_count)
        else:
            if entity_count <= 1:
                if digit_count <= self.digitcount_threshold:
                    entity = 1.5 * entity_total
                else:
                    entity = 0.5 * entity_total
            else:
                if digit_count <= self.digitcount_threshold:
                    entity = 1.5 * (entity_total / entity_count)
                else:
                    entity = 0.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 1.5 * (attribution_total + modification_total)
        else:
            attribution = 1.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 0.5 * action_total
        else:
            action = 0.5 * (action_total / action_count)
        if self.senlendiff_reverse is False and self.senlen_reverse is False:
            if diff > self.senlendiff_threshold or word_len < self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is True and self.senlen_reverse is False:
            if diff < self.senlendiff_threshold or word_len < self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is False and self.senlen_reverse is True:
            if diff > self.senlendiff_threshold or word_len > self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if diff < self.senlendiff_threshold or word_len > self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if self.jdist_reverse is False:
            if j_dist > self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if j_dist < self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.non_paraphrase_latent_space_with_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)

    def __get_latent_semantics_paraphrase_space_NLISTS_without_digitcount(self, sentence, diff, word_len, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in sentence:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if entity_count <= 1:
            entity = 1.5 * entity_total
        else:
            entity = 1.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 0.5 * (attribution_total + modification_total)
        else:
            attribution = 0.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 1.5 * action_total
        else:
            action = 1.5 * (action_total / action_count)
        if self.senlendiff_reverse is False and self.senlen_reverse is False:
            if diff <= self.senlendiff_threshold or word_len >= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is True and self.senlen_reverse is False:
            if diff >= self.senlendiff_threshold or word_len >= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is False and self.senlen_reverse is True:
            if diff <= self.senlendiff_threshold or word_len <= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if diff >= self.senlendiff_threshold or word_len <= self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if self.jdist_reverse is False:
            if j_dist <= self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if j_dist >= self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.paraphrase_latent_space_without_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)

    def __get_latent_semantics_nonparaphrase_space_NLISTS_without_digitcount(self, sentence, diff, word_len, j_dist):
        activation_function = lambda x: numpy.tanh(x)
        entity_total = numpy.array([0.0 for i in range(300)])
        attribution_total = numpy.array([0.0 for i in range(300)])
        modification_total = numpy.array([0.0 for i in range(300)])
        action_total = numpy.array([0.0 for i in range(300)])
        entity_count = 0
        attribution_count = 0
        modification_count = 0
        action_count = 0
        for word, pos in sentence:
            if pos in entity_set:
                try:
                    entity_total += self.shared_parameters.fastText[word]
                    entity_count += 1
                except:
                    continue
            elif pos in attribution_set:
                try:
                    attribution_total += self.shared_parameters.fastText[word]
                    attribution_count += 1
                except:
                    continue
            elif pos in modification_set:
                try:
                    modification_total += self.shared_parameters.fastText[word]
                    modification_count += 1
                except:
                    continue
            elif pos in action_set:
                try:
                    action_total += self.shared_parameters.fastText[word]
                    action_count += 1
                except:
                    continue
            else:
                continue
        if entity_count <= 1:
            entity = 0.5 * entity_total
        else:
            entity = 0.5 * (entity_total / entity_count)
        if (attribution_count + modification_count) <= 1:
            attribution = 1.5 * (attribution_total + modification_total)
        else:
            attribution = 1.5 * (attribution_total + modification_total) / (attribution_count + modification_count)
        if action_count <= 1:
            action = 0.5 * action_total
        else:
            action = 0.5 * (action_total / action_count)
        if self.senlendiff_reverse is False and self.senlen_reverse is False:
            if diff > self.senlendiff_threshold or word_len < self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is True and self.senlen_reverse is False:
            if diff < self.senlendiff_threshold or word_len < self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        elif self.senlendiff_reverse is False and self.senlen_reverse is True:
            if diff > self.senlendiff_threshold or word_len > self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if diff < self.senlendiff_threshold or word_len > self.senlen_threshold:
                logic = numpy.array([1.0 for i in range(300)])
            else:
                logic = 0.2 * numpy.array([1.0 for i in range(300)])
        if self.jdist_reverse is False:
            if j_dist > self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        else:
            if j_dist < self.jdist_threshold:
                jaccard = numpy.array([1.0 for i in range(300)])
            else:
                jaccard = 0.2 * numpy.array([1.0 for i in range(300)])
        word_total = numpy.concatenate((entity, attribution, action, logic, jaccard)).tolist()
        inputs_T = numpy.array(word_total, ndmin=2).T
        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.shared_parameters.non_paraphrase_latent_space_without_digitcount, self.input_holder: inputs_T})
        final_outputs = activation_function(final_inputs)
        length = len(final_outputs)
        finals = []
        for i in range(length):
            finals.append(final_outputs[i][0])
        return numpy.array(finals)

    def get_features_for_NLI_STS_tasks(self, sentence1, sentence2, with_digit_count=True):
        print("the current input sentence pair is:")
        print(sentence1)
        print(sentence2)
        print("-------------------------------------------")
        tokens1 = word_tokenize(sentence1)
        tokens2 = word_tokenize(sentence2)
        diff = abs(len(tokens1) - len(tokens2))
        word_len1 = len(tokens1)
        word_len2 = len(tokens2)
        words1 = [word.lower() for word in tokens1]
        pos_tags1 = nltk.pos_tag(words1)
        words2 = [word.lower() for word in tokens2]
        pos_tags2 = nltk.pos_tag(words2)
        words11 = []
        words22 = []

        digit_count = 0
        digit_count1 = 0
        digit_count2 = 0
        for word, pos in pos_tags1:
            if pos == 'CD':
                digit_count += 1
                digit_count1 += 1
            elif pos in entity_set:
                words11.append(word)
            elif pos in action_set:
                words11.append(word)
        for word, pos in pos_tags2:
            if pos == 'CD':
                digit_count += 1
                digit_count2 += 1
            elif pos in entity_set:
                words22.append(word)
            elif pos in action_set:
                words22.append(word)
        digit_diff = abs(digit_count1 - digit_count2)

        wmdist = self.shared_parameters.fastText.wmdistance(words11, words22)
        if math.isnan(wmdist) or math.isinf(wmdist):
            wmdist = 0

        entity_count1 = 0
        entity_count2 = 0
        action_count1 = 0
        action_count2 = 0
        for word, pos in pos_tags1:
            if pos in entity_set:
                entity_count1 += 1
            elif pos in action_set:
                action_count1 += 1
        for word, pos in pos_tags2:
            if pos in entity_set:
                entity_count2 += 1
            elif pos in action_set:
                action_count2 += 1
        entity_diff = abs(entity_count1 - entity_count2)
        action_diff = abs(action_count1 - action_count2)

        j_dist = jaccard_distance(set(words1), set(words2))

        sentence1_unigram = ngrams(words1, 1)
        sentence2_unigram = ngrams(words2, 1)
        overlap = len(set(sentence1_unigram).intersection(set(sentence2_unigram)))
        unigram_precision = overlap / word_len1
        unigram_recall = overlap / word_len2

        sentence1_bigram = ngrams(words1, 2)
        sentence2_bigram = ngrams(words2, 2)
        overlap = len(set(sentence1_bigram).intersection(set(sentence2_bigram)))
        bigram_precision = overlap / word_len1
        bigram_recall = overlap / word_len2

        sentence1_trigram = ngrams(words1, 3)
        sentence2_trigram = ngrams(words2, 3)
        overlap = len(set(sentence1_trigram).intersection(set(sentence2_trigram)))
        trigram_precision = overlap / word_len1
        trigram_recall = overlap / word_len2

        bleu_score_p1 = sentence_bleu([words1], words2, weights=(0.5, 0, 0.5, 0))
        bleu_score_r1 = sentence_bleu([words2], words1, weights=(0.5, 0, 0.5, 0))
        bleu_score_p2 = sentence_bleu([words1], words2, weights=(0, 0.5, 0, 0.5))
        bleu_score_r2 = sentence_bleu([words2], words1, weights=(0, 0.5, 0, 0.5))
        chrf1 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=5)
        chrf2 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=5)
        chrf3 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=6)
        chrf4 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=6)
        chrf7 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=4)
        chrf8 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=4)
        chrf9 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=3)
        chrf10 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=3)
        chrf11 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=2)
        chrf12 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=2)
        chrf13 = sentence_chrf(sentence1, sentence2, min_len=1, max_len=1)
        chrf14 = sentence_chrf(sentence2, sentence1, min_len=1, max_len=1)

        tl1 = []
        tl2 = []
        for word, pos in pos_tags1:
            if word in words2:
                continue
            else:
                tl1.append((word, pos))
        for word, pos in pos_tags2:
            if word in words1:
                continue
            else:
                tl2.append((word, pos))

        if with_digit_count is True:
            latent_semantics1_p = self.__get_latent_semantics_paraphrase_space_NLISTS_with_digitcount(tl1, diff, word_len1, digit_count, j_dist)
            latent_semantics2_p = self.__get_latent_semantics_paraphrase_space_NLISTS_with_digitcount(tl2, diff, word_len2, digit_count, j_dist)
            latent_semantics1_np = self.__get_latent_semantics_nonparaphrase_space_NLISTS_with_digitcount(tl1, diff, word_len1, digit_count, j_dist)
            latent_semantics2_np = self.__get_latent_semantics_nonparaphrase_space_NLISTS_with_digitcount(tl2, diff, word_len2, digit_count, j_dist)
            tmp = cos_sim(latent_semantics1_p, latent_semantics2_p)
            tmp2 = cos_sim(latent_semantics1_np, latent_semantics2_np)
            tmp3 = e_distance(latent_semantics1_p, latent_semantics1_np)
            tmp4 = e_distance(latent_semantics2_p, latent_semantics2_np)

            latent_s = numpy.concatenate(((latent_semantics1_p + latent_semantics2_p), abs(latent_semantics1_p - latent_semantics2_p), (latent_semantics1_np + latent_semantics2_np), abs(latent_semantics1_np - latent_semantics2_np))).tolist()
            latent_s.append(edit_distance(sentence1, sentence2))
            latent_s.append(wmdist)
            latent_s.append(tmp)
            latent_s.append(tmp2)
            latent_s.append(tmp3)
            latent_s.append(tmp4)
            latent_s.append(digit_diff)
            latent_s.append(entity_diff)
            latent_s.append(action_diff)
            latent_s.append(diff)
            latent_s.append(unigram_precision)
            latent_s.append(unigram_recall)
            latent_s.append(bigram_precision)
            latent_s.append(bigram_recall)
            latent_s.append(trigram_precision)
            latent_s.append(trigram_recall)
            latent_s.append(bleu_score_p1)
            latent_s.append(bleu_score_r1)
            latent_s.append(bleu_score_p2)
            latent_s.append(bleu_score_r2)
            latent_s.append(chrf1)
            latent_s.append(chrf2)
            latent_s.append(chrf3)
            latent_s.append(chrf4)
            latent_s.append(chrf7)
            latent_s.append(chrf8)
            latent_s.append(chrf9)
            latent_s.append(chrf10)
            latent_s.append(chrf11)
            latent_s.append(chrf12)
            latent_s.append(chrf13)
            latent_s.append(chrf14)
            n1 = 0
            n2 = 0
            for word in words1:
                if word in negation_set:
                    n1 += 1
            for word in words2:
                if word in negation_set:
                    n2 += 1
            latent_s.append(abs(n1 - n2))
            syn = 0
            ant = 0
            for word, pos in tl1:
                try:
                    word = self.lemmatizer.lemmatize(word)
                except:
                    continue
                synonyms = []
                antonyms = []
                for syn1 in wn.synsets(word):
                    for l in syn1.lemmas():
                        synonyms.append(l.name())
                        if l.antonyms():
                            antonyms.append(l.antonyms()[0].name())
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in synonyms:
                            syn += 1
                    except:
                        continue
                comb_name = word + ".n.01"
                try:
                    hyponyms = wn.synset(comb_name).hyponyms()
                    word_hyponyms = [lemma.name() for synset in hyponyms for lemma in synset.lemmas()]
                except:
                    print("hyponyms are not found for this word: " + word)
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in word_hyponyms:
                            syn += 1
                    except:
                        continue
                try:
                    hypernyms = wn.synset(comb_name).hypernyms()
                    word_hypernyms = [lemma.name() for synset in hypernyms for lemma in synset.lemmas()]
                except:
                    print("hypernyms are not found for this word: " + word)
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in word_hypernyms:
                            syn += 1
                    except:
                        continue
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in antonyms:
                            ant += 1
                    except:
                        continue
            latent_s.append(syn)
            latent_s.append(ant)
            latent_s.append(word_len1)
            latent_s.append(word_len2)
            latent_s.append(len(tl1))
            latent_s.append(len(tl2))

            return latent_s
        else:
            latent_semantics1_p = self.__get_latent_semantics_paraphrase_space_NLISTS_without_digitcount(tl1, diff, word_len1, j_dist)
            latent_semantics2_p = self.__get_latent_semantics_paraphrase_space_NLISTS_without_digitcount(tl2, diff, word_len2, j_dist)
            latent_semantics1_np = self.__get_latent_semantics_nonparaphrase_space_NLISTS_without_digitcount(tl1, diff, word_len1, j_dist)
            latent_semantics2_np = self.__get_latent_semantics_nonparaphrase_space_NLISTS_without_digitcount(tl2, diff, word_len2, j_dist)
            tmp = cos_sim(latent_semantics1_p, latent_semantics2_p)
            tmp2 = cos_sim(latent_semantics1_np, latent_semantics2_np)
            tmp3 = e_distance(latent_semantics1_p, latent_semantics1_np)
            tmp4 = e_distance(latent_semantics2_p, latent_semantics2_np)

            latent_s = numpy.concatenate(((latent_semantics1_p + latent_semantics2_p), abs(latent_semantics1_p - latent_semantics2_p), (latent_semantics1_np + latent_semantics2_np), abs(latent_semantics1_np - latent_semantics2_np))).tolist()
            latent_s.append(edit_distance(sentence1, sentence2))
            latent_s.append(wmdist)
            latent_s.append(tmp)
            latent_s.append(tmp2)
            latent_s.append(tmp3)
            latent_s.append(tmp4)
            latent_s.append(digit_diff)
            latent_s.append(entity_diff)
            latent_s.append(action_diff)
            latent_s.append(diff)
            latent_s.append(unigram_precision)
            latent_s.append(unigram_recall)
            latent_s.append(bigram_precision)
            latent_s.append(bigram_recall)
            latent_s.append(trigram_precision)
            latent_s.append(trigram_recall)
            latent_s.append(bleu_score_p1)
            latent_s.append(bleu_score_r1)
            latent_s.append(bleu_score_p2)
            latent_s.append(bleu_score_r2)
            latent_s.append(chrf1)
            latent_s.append(chrf2)
            latent_s.append(chrf3)
            latent_s.append(chrf4)
            latent_s.append(chrf7)
            latent_s.append(chrf8)
            latent_s.append(chrf9)
            latent_s.append(chrf10)
            latent_s.append(chrf11)
            latent_s.append(chrf12)
            latent_s.append(chrf13)
            latent_s.append(chrf14)
            n1 = 0
            n2 = 0
            for word in words1:
                if word in negation_set:
                    n1 += 1
            for word in words2:
                if word in negation_set:
                    n2 += 1
            latent_s.append(abs(n1 - n2))
            syn = 0
            ant = 0
            for word, pos in tl1:
                try:
                    word = self.lemmatizer.lemmatize(word)
                except:
                    continue
                synonyms = []
                antonyms = []
                for syn1 in wn.synsets(word):
                    for l in syn1.lemmas():
                        synonyms.append(l.name())
                        if l.antonyms():
                            antonyms.append(l.antonyms()[0].name())
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in synonyms:
                            syn += 1
                    except:
                        continue
                comb_name = word + ".n.01"
                try:
                    hyponyms = wn.synset(comb_name).hyponyms()
                    word_hyponyms = [lemma.name() for synset in hyponyms for lemma in synset.lemmas()]
                except:
                    print("hyponyms are not found for this word: " + word)
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in word_hyponyms:
                            syn += 1
                    except:
                        continue
                try:
                    hypernyms = wn.synset(comb_name).hypernyms()
                    word_hypernyms = [lemma.name() for synset in hypernyms for lemma in synset.lemmas()]
                except:
                    print("hypernyms are not found for this word: " + word)
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in word_hypernyms:
                            syn += 1
                    except:
                        continue
                for s2_word, pos in tl2:
                    try:
                        if self.lemmatizer.lemmatize(s2_word) in antonyms:
                            ant += 1
                    except:
                        continue
            latent_s.append(syn)
            latent_s.append(ant)
            latent_s.append(word_len1)
            latent_s.append(word_len2)
            latent_s.append(len(tl1))
            latent_s.append(len(tl2))

            return latent_s

    def get_paraphrase_latent_representations_for_sentence_pair(self, sentence1, sentence2, with_digit_count=True):
        print("the current input sentence pair is:")
        print(sentence1)
        print(sentence2)
        print("-------------------------------------------")
        tokens1 = word_tokenize(sentence1)
        tokens2 = word_tokenize(sentence2)
        diff = abs(len(tokens1) - len(tokens2))
        word_len1 = len(tokens1)
        word_len2 = len(tokens2)
        words1 = [word.lower() for word in tokens1]
        pos_tags1 = nltk.pos_tag(words1)
        words2 = [word.lower() for word in tokens2]
        pos_tags2 = nltk.pos_tag(words2)

        digit_count = 0
        for word, pos in pos_tags1:
            if pos == 'CD':
                digit_count += 1
        for word, pos in pos_tags2:
            if pos == 'CD':
                digit_count += 1

        j_dist = jaccard_distance(set(words1), set(words2))

        if with_digit_count is True:
            latent_semantics1_p = self.__get_latent_semantics_paraphrase_space_PI_with_digitcount(sentence1, diff, word_len1, digit_count, j_dist)
            latent_semantics2_p = self.__get_latent_semantics_paraphrase_space_PI_with_digitcount(sentence2, diff, word_len2, digit_count, j_dist)
            return (latent_semantics1_p, latent_semantics2_p)
        else:
            latent_semantics1_p = self.__get_latent_semantics_paraphrase_space_PI_without_digitcount(sentence1, diff, word_len1, j_dist)
            latent_semantics2_p = self.__get_latent_semantics_paraphrase_space_PI_without_digitcount(sentence2, diff, word_len2, j_dist)
            return (latent_semantics1_p, latent_semantics2_p)

    def get_nonparaphrase_latent_representations_for_sentence_pair(self, sentence1, sentence2, with_digit_count=True):
        print("the current input sentence pair is:")
        print(sentence1)
        print(sentence2)
        print("-------------------------------------------")
        tokens1 = word_tokenize(sentence1)
        tokens2 = word_tokenize(sentence2)
        diff = abs(len(tokens1) - len(tokens2))
        word_len1 = len(tokens1)
        word_len2 = len(tokens2)
        words1 = [word.lower() for word in tokens1]
        pos_tags1 = nltk.pos_tag(words1)
        words2 = [word.lower() for word in tokens2]
        pos_tags2 = nltk.pos_tag(words2)

        digit_count = 0
        for word, pos in pos_tags1:
            if pos == 'CD':
                digit_count += 1
        for word, pos in pos_tags2:
            if pos == 'CD':
                digit_count += 1

        j_dist = jaccard_distance(set(words1), set(words2))

        if with_digit_count is True:
            latent_semantics1_np = self.__get_latent_semantics_nonparaphrase_space_PI_with_digitcount(sentence1, diff, word_len1, digit_count, j_dist)
            latent_semantics2_np = self.__get_latent_semantics_nonparaphrase_space_PI_with_digitcount(sentence2, diff, word_len2, digit_count, j_dist)
            return (latent_semantics1_np, latent_semantics2_np)
        else:
            latent_semantics1_np = self.__get_latent_semantics_nonparaphrase_space_PI_without_digitcount(sentence1, diff, word_len1, j_dist)
            latent_semantics2_np = self.__get_latent_semantics_nonparaphrase_space_PI_without_digitcount(sentence2, diff, word_len2, j_dist)
            return (latent_semantics1_np, latent_semantics2_np)


class CharacteristicsDetector:

    def __init__(self, sentence_pair_list, label_list):
        self.sentence_pair_list = sentence_pair_list
        self.label_list = label_list

    def detect(self, histogram_bins=100):
        jaccard1 = []
        jaccard2 = []
        sent_len1 = []
        sent_len2 = []
        sent_diff1 = []
        sent_diff2 = []
        dsum1 = []
        dsum2 = []

        index = 0
        paraphrase_sentence_pairs = 0
        non_paraphrase_sentence_pairs = 0
        for sentence1, sentence2 in self.sentence_pair_list:
            print("the current sentence pair being dectected is:")
            print(sentence1)
            print(sentence2)
            print("-------------------------------------------")

            tokens1 = word_tokenize(sentence1)
            tokens2 = word_tokenize(sentence2)
            diff = abs(len(tokens1) - len(tokens2))
            words1 = [word.lower() for word in tokens1]
            pos_tags1 = nltk.pos_tag(words1)
            words2 = [word.lower() for word in tokens2]
            pos_tags2 = nltk.pos_tag(words2)

            if self.label_list[index] == int(1):
                sent_diff1.append(diff)
                sent_len1.append(len(tokens1))
                sent_len1.append(len(tokens2))
                paraphrase_sentence_pairs += 1
            else:
                sent_diff2.append(diff)
                sent_len2.append(len(tokens1))
                sent_len2.append(len(tokens2))
                non_paraphrase_sentence_pairs += 1

            sum = 0
            sum1 = 0
            sum2 = 0
            for word, pos in pos_tags1:
                if pos == 'CD':
                    sum += 1
                    sum1 += 1
            for word, pos in pos_tags2:
                if pos == 'CD':
                    sum += 1
                    sum2 += 1

            if sum > 0:
                if self.label_list[index] == int(1):
                    dsum1.append(sum)
                else:
                    dsum2.append(sum)

            j_dist = jaccard_distance(set(words1), set(words2))
            if self.label_list[index] == int(1):
                jaccard1.append(j_dist)
            else:
                jaccard2.append(j_dist)

            index += 1

        seaborn.distplot(jaccard1, hist=True, kde=True, bins=histogram_bins, kde_kws={"label": "paraphrase"})
        seaborn.distplot(jaccard2, hist=True, kde=True, bins=histogram_bins, kde_kws={"label": "non-paraphrase"})
        plt.xlabel("Jaccard distance")
        plt.ylabel("Number of pairs")
        plt.show()

        seaborn.distplot(dsum1, hist=True, kde=True, bins=histogram_bins, kde_kws={"label": "paraphrase"})
        seaborn.distplot(dsum2, hist=True, kde=True, bins=histogram_bins, kde_kws={"label": "non-paraphrase"})
        plt.xlabel("digit-count")
        plt.ylabel("Number of pairs")
        plt.show()

        seaborn.distplot(sent_diff1, hist=True, kde=True, bins=histogram_bins, kde_kws={"label": "paraphrase"})
        seaborn.distplot(sent_diff2, hist=True, kde=True, bins=histogram_bins, kde_kws={"label": "non-paraphrase"})
        plt.xlabel("sentence length difference")
        plt.ylabel("Number of pairs")
        plt.show()

        seaborn.distplot(sent_len1, hist=True, kde=True, bins=histogram_bins, kde_kws={"label": "paraphrase"})
        seaborn.distplot(sent_len2, hist=True, kde=True, bins=histogram_bins, kde_kws={"label": "non-paraphrase"})
        plt.xlabel("sentence length")
        plt.ylabel("Number of pairs")
        plt.show()

        print("total paraphrase sentence pairs are: " + str(paraphrase_sentence_pairs))
        print("total non-paraphrase sentence pairs are: " + str(non_paraphrase_sentence_pairs))