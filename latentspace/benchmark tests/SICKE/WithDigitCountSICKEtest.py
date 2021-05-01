import torch
import latentspace
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=439, out_features=900)
        self.layer2 = torch.nn.Linear(in_features=900, out_features=100)
        self.layer3 = torch.nn.Linear(in_features=100, out_features=3)

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
            if t[0] > t[1] and t[0] > t[2]:
                ans.append(0)
            elif t[2] > t[0] and t[2] > t[1]:
                ans.append(2)
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
        if all_values[3] == "ENTAILMENT":
            test_label_list.append(int(2))
        elif all_values[3] == "NEUTRAL":
            test_label_list.append(int(1))
        else:
            test_label_list.append(int(0))

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

    true_preds = 0
    sum = 0

    for batch in v_dataloader:
        b_inputs = batch[0].to(device)
        b_labels = batch[1].to(device)

        with torch.no_grad():
            predcited_labels = classifier.predict(b_inputs)
            outputs = classifier(b_inputs)

        # Move logits and labels to CPU
        logits = predcited_labels.detach().cpu().numpy().tolist()
        label_ids = b_labels.to('cpu').numpy().tolist()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        predicts, t_sum = flat_accuracy(logits, label_ids)
        true_preds += predicts
        sum += t_sum

    # Report the final accuracy for this test run.
    avg_val_accuracy = true_preds / sum
    print("  Accuracy: {0:.3f}".format(avg_val_accuracy))