from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy



if __name__ == "__main__":

    paraphrase_file = open(r"directory/msr_paraphrase_train_1.csv", 'r', encoding='utf-16')
    paraphrase_list = paraphrase_file.readlines()
    paraphrase_file.close()
    non_paraphrase_file = open(r"directory/msr_paraphrase_train_0.csv", 'r', encoding='utf-16')
    non_paraphrase_list = non_paraphrase_file.readlines()
    non_paraphrase_file.close()

    documents = []
    description_counts = []

    for record in paraphrase_list:
        record = record.rstrip("\n")
        all_values = record.split('*')
        all_values = all_values[1:]

        count = 0
        for sentence in all_values:
            if len(sentence) == 0:
                continue
            count += 1
            documents.append(sentence)
        description_counts.append(count)

    print(len(description_counts))

    for record in non_paraphrase_list:
        record = record.rstrip("\n")
        all_values = record.split('*')
        all_values = all_values[1:]

        for sentence in all_values:
            if len(sentence) == 0:
                continue
            documents.append(sentence)
            description_counts.append(1)

    print(len(description_counts))


    cv = CountVectorizer()
    term_document_matrix = cv.fit_transform(documents).toarray()
    print(term_document_matrix.shape)

    svd = TruncatedSVD(n_components=100)
    svd_matrix = svd.fit_transform(term_document_matrix)
    print(svd_matrix.shape)

    labels = []
    index = 0
    for count in description_counts:
        sum = 0
        for i in range(count):
            sum += (svd_matrix[index] / 5.6)
            index += 1
        average = sum / count
        labels.append(average)

    ### test if the generated latent semantics are identical to original singular vectors
    tmp = (svd_matrix[0] + svd_matrix[1]) / 11.2
    similarity = cosine_similarity([svd_matrix[0] / 5.6], [tmp])
    print(similarity)
    print(cosine_similarity([svd_matrix[0] / 5.6], [labels[0]]))

    matrix = numpy.matrix(labels)
    dataframe = pd.DataFrame(data=matrix.astype(float))
    dataframe.to_csv(r'directory/latent_representation_100d.csv', sep='*', header=False, index=False)



