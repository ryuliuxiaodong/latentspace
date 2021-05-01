from gensim.models.keyedvectors import KeyedVectors
import tensorflow as tf
import numpy
import math
from nltk.tokenize import word_tokenize
import nltk
from nltk.metrics import *
import matplotlib.pyplot as plt


with tf.device("/device:gpu:0"):
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
matrix_holder = tf.placeholder(tf.float32)
input_holder = tf.placeholder(tf.float32)
final_inputs_holder = tf.matmul(matrix_holder, input_holder)

entity_set = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP']
modification_set = ['DT', 'PDT', 'JJ', 'JJR', 'JJS', 'PRP$', 'POS', 'EX', 'MD', 'RB', 'RBR', 'RBS', 'RP', 'TO']
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

def get_latent_semantics_paraphrase_space(fasttext, matrix, sentence, diff, word_len, j_dist):
    activation_function = lambda x: numpy.tanh(x)
    tokens = word_tokenize(sentence)
    if len(tokens) == 0:
        return 0
    words = [word.lower() for word in tokens]
    pos_tags = nltk.pos_tag(words)
    entity_total = numpy.array([0.0 for i in range(300)])
    modification_total = numpy.array([0.0 for i in range(300)])
    action_total = numpy.array([0.0 for i in range(300)])
    entity_count = 0
    modification_count = 0
    action_count = 0
    for word, pos in pos_tags:
        if pos in entity_set:
            try:
                entity_total += fasttext[word]
                entity_count += 1
            except:
                continue
        elif pos in modification_set:
            try:
                modification_total += fasttext[word]
                modification_count += 1
            except:
                continue
        elif pos in action_set:
            try:
                action_total += fasttext[word]
                action_count += 1
            except:
                continue
        else:
            continue
    if entity_count <= 1:
        entity = 1.5 * entity_total
    else:
        entity = 1.5 * (entity_total / entity_count)
    if modification_count <= 1:
        modification = 0.5 * modification_total
    else:
        modification = 0.5 * (modification_total / modification_count)
    if action_count <= 1:
        action = 1.5 * action_total
    else:
        action = 1.5 * (action_total / action_count)
    if diff <= 5 or word_len >= 23:
        RF1 = numpy.array([1.0 for i in range(300)])
    else:
        RF1 = 0.2 * numpy.array([1.0 for i in range(300)])
    if j_dist <= 0.6:
        RF2 = numpy.array([1.0 for i in range(300)])
    else:
        RF2 = 0.2 * numpy.array([1.0 for i in range(300)])
    word_total = numpy.concatenate((entity, modification, action, RF1, RF2)).tolist()
    inputs_T = numpy.array(word_total, ndmin=2).T
    final_inputs = sess.run(final_inputs_holder, {matrix_holder: matrix, input_holder: inputs_T})
    final_outputs = activation_function(final_inputs)
    length = len(final_outputs)
    finals = []
    for i in range(length):
        finals.append(final_outputs[i][0])
    return numpy.array(finals)


def get_latent_semantics_nonparaphrase_space(fasttext, matrix, sentence, diff, word_len, j_dist):
    activation_function = lambda x: numpy.tanh(x)
    tokens = word_tokenize(sentence)
    if len(tokens) == 0:
        return 0
    words = [word.lower() for word in tokens]
    pos_tags = nltk.pos_tag(words)
    entity_total = numpy.array([0.0 for i in range(300)])
    modification_total = numpy.array([0.0 for i in range(300)])
    action_total = numpy.array([0.0 for i in range(300)])
    entity_count = 0
    modification_count = 0
    action_count = 0
    for word, pos in pos_tags:
        if pos in entity_set:
            try:
                entity_total += fasttext[word]
                entity_count += 1
            except:
                continue
        elif pos in modification_set:
            try:
                modification_total += fasttext[word]
                modification_count += 1
            except:
                continue
        elif pos in action_set:
            try:
                action_total += fasttext[word]
                action_count += 1
            except:
                continue
        else:
            continue
    if entity_count <= 1:
        entity = 0.5 * entity_total
    else:
        entity = 0.5 * (entity_total / entity_count)
    if modification_count <= 1:
        modification = 1.5 * modification_total
    else:
        modification = 1.5 * (modification_total / modification_count)
    if action_count <= 1:
        action = 0.5 * action_total
    else:
        action = 0.5 * (action_total / action_count)
    if diff > 5 or word_len < 23:
        RF1 = numpy.array([1.0 for i in range(300)])
    else:
        RF1 = 0.2 * numpy.array([1.0 for i in range(300)])
    if j_dist > 0.6:
        RF2 = numpy.array([1.0 for i in range(300)])
    else:
        RF2 = 0.2 * numpy.array([1.0 for i in range(300)])
    word_total = numpy.concatenate((entity, modification, action, RF1, RF2)).tolist()
    inputs_T = numpy.array(word_total, ndmin=2).T
    final_inputs = sess.run(final_inputs_holder, {matrix_holder: matrix, input_holder: inputs_T})
    final_outputs = activation_function(final_inputs)
    length = len(final_outputs)
    finals = []
    for i in range(length):
        finals.append(final_outputs[i][0])
    return numpy.array(finals)

if __name__ == "__main__":
    paraphrase_file = open(r"D:\Dataset\paraphrase\English\MSRParaphraseCorpus\msr_paraphrase_train_1.csv", 'r', encoding='utf-16')
    paraphrase_list = paraphrase_file.readlines()
    paraphrase_file.close()
    non_paraphrase_file = open(r"D:\Dataset\paraphrase\English\MSRParaphraseCorpus\msr_paraphrase_train_0.csv", 'r', encoding='utf-16')
    non_paraphrase_list = non_paraphrase_file.readlines()
    non_paraphrase_file.close()

    paraphrase_sentence_pairs = []
    nonparaphrase_sentence_pairs = []
    for record in paraphrase_list:
        record = record.rstrip("\n")
        all_values = record.split('*')
        all_values = all_values[1:]
        sentence1 = all_values[0]
        sentence2 = all_values[1]
        paraphrase_sentence_pairs.append((sentence1, sentence2))
    for record in non_paraphrase_list:
        record = record.rstrip("\n")
        all_values = record.split('*')
        all_values = all_values[1:]
        sentence1 = all_values[0]
        sentence2 = all_values[1]
        nonparaphrase_sentence_pairs.append((sentence1, sentence2))

    fasttext = KeyedVectors.load_word2vec_format("D:\Model\pretrained\FastText\crawl-300d-2M-subword.vec", binary=False)

    flag = 50
    cos_diff = []
    x_axis = []

    while flag <= 800:
        print(flag)
        x_axis.append(flag)

        matrix_file = open(r'D:\Model\pretrained\latentspaes\wo\paraphrase_latent_space_without_digitcount_at_epochs_' + str(flag) + '.csv', 'r')
        matrix_table = matrix_file.readlines()
        matrix_file.close()
        matrix_file = open(r'D:\Model\pretrained\latentspaes\wo\non-paraphrase_latent_space_without_digitcount_at_epochs_' + str(flag) + '.csv', 'r')
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
        matrix = matrix_list
        for record in matrix_table2:
            tmp = []
            record = record.rstrip("\n")
            all_values = record.split('*')
            for value in all_values:
                tmp.append(float(value))
            matrix_list2.append(tmp)
        matrix2 = matrix_list2

        cos_sim_paraphrase_space = []
        cos_sim_nonparaphrase_space = []
        for sentence1, sentence2 in paraphrase_sentence_pairs:
            tokens1 = word_tokenize(sentence1)
            tokens2 = word_tokenize(sentence2)
            diff = abs(len(tokens1) - len(tokens2))

            word_len1 = len(tokens1)
            word_len2 = len(tokens2)

            words1 = [word.lower() for word in tokens1]
            pos_tags1 = nltk.pos_tag(words1)
            words2 = [word.lower() for word in tokens2]
            pos_tags2 = nltk.pos_tag(words2)

            count = 0
            for word, pos in pos_tags1:
                if pos == 'CD':
                    count += 1
            for word, pos in pos_tags2:
                if pos == 'CD':
                    count += 1

            j_dist = jaccard_distance(set(words1), set(words2))

            latent_semantics1_p = get_latent_semantics_paraphrase_space(fasttext, matrix, sentence1, diff, word_len1, j_dist)
            latent_semantics2_p = get_latent_semantics_paraphrase_space(fasttext, matrix, sentence2, diff, word_len2, j_dist)
            latent_semantics1_np = get_latent_semantics_nonparaphrase_space(fasttext, matrix2, sentence1, diff, word_len1, j_dist)
            latent_semantics2_np = get_latent_semantics_nonparaphrase_space(fasttext, matrix2, sentence2, diff, word_len2, j_dist)
            tmp = cos_sim(latent_semantics1_p, latent_semantics2_p)
            tmp2 = cos_sim(latent_semantics1_np, latent_semantics2_np)

            cos_sim_paraphrase_space.append(tmp)
            cos_sim_nonparaphrase_space.append(tmp2)

        diff1 = numpy.mean(cos_sim_nonparaphrase_space) - numpy.mean(cos_sim_paraphrase_space)

        cos_sim_paraphrase_space = []
        cos_sim_nonparaphrase_space = []
        for sentence1, sentence2 in nonparaphrase_sentence_pairs:
            tokens1 = word_tokenize(sentence1)
            tokens2 = word_tokenize(sentence2)
            diff = abs(len(tokens1) - len(tokens2))

            word_len1 = len(tokens1)
            word_len2 = len(tokens2)

            words1 = [word.lower() for word in tokens1]
            pos_tags1 = nltk.pos_tag(words1)
            words2 = [word.lower() for word in tokens2]
            pos_tags2 = nltk.pos_tag(words2)

            count = 0
            for word, pos in pos_tags1:
                if pos == 'CD':
                    count += 1
            for word, pos in pos_tags2:
                if pos == 'CD':
                    count += 1

            j_dist = jaccard_distance(set(words1), set(words2))

            latent_semantics1_p = get_latent_semantics_paraphrase_space(fasttext, matrix, sentence1, diff, word_len1, j_dist)
            latent_semantics2_p = get_latent_semantics_paraphrase_space(fasttext, matrix, sentence2, diff, word_len2, j_dist)
            latent_semantics1_np = get_latent_semantics_nonparaphrase_space(fasttext, matrix2, sentence1, diff, word_len1, j_dist)
            latent_semantics2_np = get_latent_semantics_nonparaphrase_space(fasttext, matrix2, sentence2, diff, word_len2, j_dist)
            tmp = cos_sim(latent_semantics1_p, latent_semantics2_p)
            tmp2 = cos_sim(latent_semantics1_np, latent_semantics2_np)

            cos_sim_paraphrase_space.append(tmp)
            cos_sim_nonparaphrase_space.append(tmp2)

        diff2 = numpy.mean(cos_sim_nonparaphrase_space) - numpy.mean(cos_sim_paraphrase_space)

        cos_diff.append(diff2 - diff1)

        flag += 50

    fig, ax = plt.subplots()
    x = x_axis
    y = cos_diff
    group_labels = ['50', '100', '150', '200', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '800']
    plt.plot(x, y)
    plt.xticks(x, group_labels, rotation=0)
    plt.grid()
    plt.title('without digit-count scheme')
    plt.xlabel('epochs')
    plt.ylabel('difference of discriminative similarity')
    plt.show()