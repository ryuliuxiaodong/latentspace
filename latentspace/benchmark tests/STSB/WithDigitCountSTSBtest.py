import torch
import latentspace
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from scipy import stats


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=439, out_features=900)
        self.layer2 = torch.nn.Linear(in_features=900, out_features=100)
        self.layer3 = torch.nn.Linear(in_features=100, out_features=1)

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

def combine(big_ls, small_ls):
    length = len(small_ls)
    for i in range(length):
        big_ls.append(small_ls[i][0])

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
    test_file = open(r"/sts-test.csv", 'r', encoding='utf-8')
    test_list = test_file.readlines()
    test_file.close()
    for record in test_list:
        record = record.rstrip("\n")
        all_values = record.split('\t')
        sentence1 = all_values[5]
        sentence2 = all_values[6]
        label = all_values[4]
        test_sentence_pair_list.append((sentence1, sentence2))
        test_label_list.append(float(label) / 5.0)

    feature_list = []

    for sent1, sent2 in test_sentence_pair_list:
        latent_s = defaultSettings.get_features_for_NLI_STS_tasks(sent1, sent2)
        feature_list.append(latent_s)

    v_inputs = torch.tensor(feature_list).float()
    v_labels = torch.tensor(test_label_list)
    v_dataset = TensorDataset(v_inputs, v_labels)

    v_dataloader = DataLoader(
        v_dataset,  # The training samples.
        sampler=SequentialSampler(v_dataset),  # Select batches randomly
        batch_size=32  # Trains with this batch size.
    )

    classifier.cuda()
    classifier.eval()

    total_pred = []

    for batch in v_dataloader:
        b_inputs = batch[0].to(device)
        b_labels = batch[1].to(device)

        with torch.no_grad():
            predcited_labels = classifier.predict(b_inputs)

        logits = predcited_labels.detach().cpu().numpy().tolist()
        label_ids = b_labels.to('cpu').numpy().tolist()

        combine(total_pred, logits)

    pearsonr = stats.pearsonr(total_pred, test_label_list)[0]
    print("  Accuracy: {0:.3f}".format(pearsonr))
    spearmanr = stats.spearmanr(total_pred, test_label_list)[0]
    print("  Accuracy: {0:.3f}".format(spearmanr))

    print("  Total predicted data: " + str(len(total_pred)))