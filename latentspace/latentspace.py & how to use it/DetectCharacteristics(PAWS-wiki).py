import latentspace

if __name__ == "__main__":
    train_file = open(r"/PAWSX/en/train.tsv", 'r', encoding='utf-8')
    train_list = train_file.readlines()
    train_file.close()

    sentence_pair_list = []
    label_list = []
    for record in train_list:
        record = record.rstrip("\n")
        all_values = record.split('\t')
        if all_values[3] == "label":
            continue
        sentence1 = all_values[1]
        sentence2 = all_values[2]
        if int(all_values[3]) == 1:
            sentence_pair_list.append((sentence1, sentence2))
            label_list.append(int(1))
        else:
            sentence_pair_list.append((sentence1, sentence2))
            label_list.append(int(0))


    detector = latentspace.CharacteristicsDetector(sentence_pair_list, label_list)
    detector.detect(histogram_bins=100)