import torch
import latentspace
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from scipy import stats


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=439, out_features=900)
        self.layer2 = torch.nn.Linear(in_features=900, out_features=100)
        self.layer3 = torch.nn.Linear(in_features=100, out_features=2)

    def forward(self, input):
        x = self.layer1(input)
        y = torch.nn.functional.relu(x)
        z = self.layer2(y)
        zz = torch.nn.functional.relu(z)
        zzz = self.layer3(zz)
        zzzz = torch.nn.functional.sigmoid(zzz)
        return zzzz

    def predict(self, input):
        return self.forward(input)


if __name__ == "__main__":

    device = "/device:gpu:0"
    fastText_file = "/crawl-300d-2M-subword.vec"
    paraphrase_latent_space_with_digitcount_file = r'/paraphrase_latent_space_with_digitcount.csv'
    non_paraphrase_latent_space_with_digitcount_file = r'/nonparaphrase_latent_space_with_digitcount.csv'
    paraphrase_latent_space_without_digitcount_file = r'/paraphrase_latent_space_without_digitcount.csv'
    non_paraphrase_latent_space_without_digitcount_file = r'/nonparaphrase_latent_space_without_digitcount.csv'
    sharedParameters = latentspace.SharedParameters(fastText_file, paraphrase_latent_space_with_digitcount_file, non_paraphrase_latent_space_with_digitcount_file, paraphrase_latent_space_without_digitcount_file, non_paraphrase_latent_space_without_digitcount_file)
    defaultSettings = latentspace.DefaultSettings(device, sharedParameters)

    classifier = torch.load(r"/tuned classifiers/default settings/without digit-count/classifier.pth")
    device = torch.device('cuda')

    test_sentence_pair_list = []
    test_label_list = []
    test_file = open(r"/SICK.txt", 'r', encoding='utf-16')
    test_list = test_file.readlines()
    test_file.close()
    for record in test_list:
        record = record.rstrip("\n")
        all_values = record.split('\t')
        if all_values[11] != "TEST":
            continue
        sentence1 = all_values[1]
        sentence2 = all_values[2]
        test_sentence_pair_list.append((sentence1, sentence2))
        label = float(all_values[4]) / 5.0
        test_label_list.append(label)

    test_feature_list_default_settings_without_digitcount = []
    for sent1, sent2 in test_sentence_pair_list:
        latent_s = defaultSettings.get_features_for_NLI_STS_tasks(sent1, sent2, with_digit_count=False)
        test_feature_list_default_settings_without_digitcount.append(latent_s)

    classifier.cuda()
    classifier.eval()

    total_predict_labels = []

    for latent_s in test_feature_list_default_settings_without_digitcount:
        score_dist = classifier(torch.FloatTensor([latent_s]).to(device)).detach().cpu().numpy().tolist()[0]
        score = score_dist[0]

        total_predict_labels.append(score)

    pearsonr = stats.pearsonr(total_predict_labels, test_label_list)[0]
    print("  pearsonr: {0:.3f}".format(pearsonr))
    spearmanr = stats.spearmanr(total_predict_labels, test_label_list)[0]
    print("  spearmanr: {0:.3f}".format(spearmanr))