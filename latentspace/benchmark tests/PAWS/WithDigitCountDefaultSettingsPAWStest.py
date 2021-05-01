import numpy
import math
import torch
import sklearn
import latentspace
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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

def flat_accuracy(preds, labels):
    sum = len(labels)
    count = 0
    for i in range(sum):
        if preds[i] == labels[i]:
            count += 1
        else:
            continue
    return (count, sum)

def softmax(x):
    return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)


if __name__ == "__main__":

    device = "/device:gpu:0"
    fastText_file = "/crawl-300d-2M-subword.vec"
    paraphrase_latent_space_with_digitcount_file = r'/paraphrase_latent_space_with_digitcount.csv'
    non_paraphrase_latent_space_with_digitcount_file = r'/nonparaphrase_latent_space_with_digitcount.csv'
    paraphrase_latent_space_without_digitcount_file = r'/paraphrase_latent_space_without_digitcount.csv'
    non_paraphrase_latent_space_without_digitcount_file = r'/nonparaphrase_latent_space_without_digitcount.csv'
    sharedParameters = latentspace.SharedParameters(fastText_file, paraphrase_latent_space_with_digitcount_file, non_paraphrase_latent_space_with_digitcount_file, paraphrase_latent_space_without_digitcount_file, non_paraphrase_latent_space_without_digitcount_file)
    defaultSettings = latentspace.DefaultSettings(device, sharedParameters)

    classifier = torch.load(r"/tuned classifiers/default settings/with digit-count/classifier.pth")
    device = torch.device('cuda')

    test_sentence_pair_list = []
    test_label_list = []
    test_file = open(r"/PAWSX/en/test_2k.tsv", 'r', encoding='utf-8')
    test_list = test_file.readlines()
    test_file.close()
    for record in test_list:
        record = record.rstrip("\n")
        all_values = record.split('\t')
        if all_values[3] == "label":
            continue
        sentence1 = all_values[1]
        sentence2 = all_values[2]
        test_sentence_pair_list.append((sentence1, sentence2))
        test_label_list.append(int(all_values[3]))

    classifier.cuda()
    classifier.eval()

    pred_labels = []
    positive_scores = []

    for sent1, sent2 in test_sentence_pair_list:

        latent_s = defaultSettings.get_features_for_PI_task(sent1, sent2)

        predict_label = classifier.predict(torch.FloatTensor([latent_s]).to(device)).detach().cpu().numpy().tolist()[0]
        score = classifier(torch.FloatTensor([latent_s]).to(device)).detach().cpu().numpy().tolist()[0]
        score = numpy.array(score)
        score = softmax(score)

        pred_labels.append(predict_label)
        positive_scores.append(score[1])

    accuracy = accuracy_score(test_label_list, pred_labels)
    print("  Accuracy: {0:.3f}".format(accuracy))
    f1s = f1_score(test_label_list, pred_labels)
    print("  F1: {0:.3f}".format(f1s))
    p = precision_score(test_label_list, pred_labels)
    print("  P: {0:.3f}".format(p))
    r = recall_score(test_label_list, pred_labels)
    print("  R: {0:.3f}".format(r))
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_label_list, positive_scores, pos_label=1)
    print("  AUC: {0:.3f}".format(sklearn.metrics.auc(fpr, tpr)))

