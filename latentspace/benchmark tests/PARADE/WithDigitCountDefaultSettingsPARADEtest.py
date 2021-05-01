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
    defaultSettings = latentspace.DefaultSettings(device, sharedParameters)

    classifier = torch.load(r"/tuned classifiers/default settings/with digit-count/classifier.pth")
    device = torch.device('cuda')

    test_sentence_pair_list = []
    test_label_list = []
    test_file = open(r"/PARADE_test.txt", 'r', encoding='utf-8')
    test_list = test_file.readlines()
    test_file.close()
    for record in test_list:
        record = record.rstrip("\n")
        all_values = record.split('\t')
        if all_values[1] == "Binary labels":
            continue
        sentence1 = all_values[3]
        sentence2 = all_values[4]
        if int(all_values[1]) == 1:
            test_sentence_pair_list.append((sentence1, sentence2))
            test_label_list.append(int(1))
        else:
            test_sentence_pair_list.append((sentence1, sentence2))
            test_label_list.append(int(0))

    classifier.cuda()
    classifier.eval()

    pred_labels = []

    for sent1, sent2 in test_sentence_pair_list:
        latent_s = defaultSettings.get_features_for_PI_task(sent1, sent2)

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