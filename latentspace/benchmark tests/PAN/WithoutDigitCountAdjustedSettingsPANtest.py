import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import latentspace


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=432, out_features=900)
        self.layer2 = torch.nn.Linear(in_features=900, out_features=100)
        self.layer3 = torch.nn.Linear(in_features=100, out_features=2)

    def forward(self, input):
        x = self.layer1(input)
        y = torch.nn.functional.relu(x)
        z = self.layer2(y)
        zz = torch.nn.functional.relu(z)
        zzz = self.layer3(zz)
        return zzz

    def predict(self, input):
        pred = torch.nn.functional.softmax(self.forward(input), dim=1)
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

if __name__ == "__main__":

    device = "/device:gpu:0"
    fastText_file = "/crawl-300d-2M-subword.vec"
    paraphrase_latent_space_with_digitcount_file = r'/paraphrase_latent_space_with_digitcount.csv'
    non_paraphrase_latent_space_with_digitcount_file = r'/nonparaphrase_latent_space_with_digitcount.csv'
    paraphrase_latent_space_without_digitcount_file = r'/paraphrase_latent_space_without_digitcount.csv'
    non_paraphrase_latent_space_without_digitcount_file = r'/nonparaphrase_latent_space_without_digitcount.csv'
    sharedParameters = latentspace.SharedParameters(fastText_file, paraphrase_latent_space_with_digitcount_file, non_paraphrase_latent_space_with_digitcount_file, paraphrase_latent_space_without_digitcount_file, non_paraphrase_latent_space_without_digitcount_file)
    adjustedSettings = latentspace.AdjustedSettings(device, sharedParameters)
    adjustedSettings.adjust_settings(senlendiff_threshold=10, senlen_threshold=30, senlen_reverse=True, jdist_threshold=0.7)

    classifier = torch.load(r"/tuned classifiers/adjusted settings/without digit-count/classifier.pth")
    device = torch.device('cuda')

    test_sentence_pair_list = []
    test_label_list = []
    test_file = open(r"pan-test_s1.txt", 'r', encoding='utf-8')
    test_list1 = test_file.readlines()
    test_file.close()
    test_file = open(r"pan-test_s2.txt", 'r', encoding='utf-8')
    test_list2 = test_file.readlines()
    test_file.close()
    test_file = open(r"pan-test.labels", 'r', encoding='utf-8')
    label_list = test_file.readlines()
    test_file.close()
    sentence1_list = []
    sentence2_list = []
    labels = []
    for record in test_list1:
        record = record.rstrip("\n")
        all_values = record.split('\t')
        sentence1_list.append(all_values[0])
    for record in test_list2:
        record = record.rstrip("\n")
        all_values = record.split('\t')
        sentence2_list.append(all_values[0])
    for record in label_list:
        record = record.rstrip("\n")
        all_values = record.split('\t')
        labels.append(int(all_values[0]))
    length = len(sentence1_list)
    for index in range(length):
        test_sentence_pair_list.append((sentence1_list[index], sentence2_list[index]))
        test_label_list.append(labels[index])

    classifier.cuda()
    classifier.eval()

    pred_labels = []

    for sent1, sent2 in test_sentence_pair_list:
        latent_s = adjustedSettings.get_features_for_PI_task(sent1, sent2, with_digit_count=False)

        predict_label = classifier.predict(torch.FloatTensor([latent_s]).to(device)).detach().cpu().numpy().tolist()[0]
        pred_labels.append(predict_label)

    accuracy = accuracy_score(test_label_list, pred_labels)
    print("  Accuracy: {0:.3f}".format(accuracy))
    f1s = f1_score(test_label_list, pred_labels)
    print("  F1: {0:.3f}".format(f1s))
    p = precision_score(test_label_list, pred_labels)
    print("  P: {0:.3f}".format(p))
    r = recall_score(test_label_list, pred_labels)
    print("  R: {0:.3f}".format(r))