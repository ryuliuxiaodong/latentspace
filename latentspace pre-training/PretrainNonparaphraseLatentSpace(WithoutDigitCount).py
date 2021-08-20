import numpy
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import bowtf
from gensim.models.keyedvectors import KeyedVectors
import nltk
from nltk.metrics import *


if __name__ == "__main__":

    fasttext = KeyedVectors.load_word2vec_format("/crawl-300d-2M-subword.vec", binary=False)

    ### initialize non-paraphrase latent space (projection matrix) with the length of input layer, output layer and learning rate
    bow2 = bowtf.BOWTF(1500, 100, 0.0005)

    non_paraphrase_file = open(r"/msr_paraphrase_train_0.csv", 'r', encoding='utf-16')
    non_paraphrase_list = non_paraphrase_file.readlines()
    non_paraphrase_file.close()

    ### load latent representations
    target_file = open(r'/MRPC_Target_normalized_100d.csv', 'r')
    target_list = target_file.readlines()
    target_file.close()

    ### initialize three sets
    entity_set = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP']
    modification_set = ['DT', 'PDT', 'JJ', 'JJR', 'JJS', 'PRP$', 'POS', 'EX', 'MD', 'RB', 'RBR', 'RBS', 'RP', 'TO']
    action_set = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    epochs = 800
    errors2 = []

    ### train non-paraphrase space
    for e in range(epochs):
        target_index = 2753
        for record in non_paraphrase_list:
            record = record.rstrip("\n")
            all_values = record.split('*')
            all_values = all_values[1:]

            tokens1 = word_tokenize(all_values[0])
            tokens2 = word_tokenize(all_values[1])
            diff = abs(len(tokens1) - len(tokens2))

            words1 = [word.lower() for word in tokens1]
            pos_tags1 = nltk.pos_tag(words1)
            words2 = [word.lower() for word in tokens2]
            pos_tags2 = nltk.pos_tag(words2)

            sum = 0
            for word, pos in pos_tags1:
                if pos == 'CD':
                    sum += 1
            for word, pos in pos_tags2:
                if pos == 'CD':
                    sum += 1

            j_dist = jaccard_distance(set(words1), set(words2))

            for sentence in all_values:
                target_values = target_list[target_index].split('*')
                target = []
                for value in target_values:
                    target.append(float(value))
                tokens = word_tokenize(sentence)
                word_len = len(tokens)
                print(e)
                print(target_index)
                print(sentence)
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
                bow2.train(word_total, target)
                target_index += 1

        ### calculate the error loss of each epoch and print out them
        error2 = bow2.error_avg()
        print(error2)
        ### collect the error loss of each epoch for final visualization
        errors2.append(error2)

        if (e+1) % 50 == 0:
            ### save the space (projection matrix) as csv files
            bow2.save_matrix_to_csv(r'/non-paraphrase_latent_space_without_digitcount_at_epochs_' + str(e+1) + '.csv')

    ### visualize the error loss of all epoches
    ax = plt.subplot()
    positions = numpy.arange(1, epochs + 1)
    ax.bar(positions, errors2, 0.5)
    plt.show()