import torch
import latentspace
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import time
import datetime
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from scipy import stats


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

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
    ###initialize SharedParameters: pre-trained fastText and latent spaces
    fastText_file = "/crawl-300d-2M-subword.vec"
    paraphrase_latent_space_with_digitcount_file = r'/paraphrase_latent_space_with_digitcount.csv'
    non_paraphrase_latent_space_with_digitcount_file = r'/nonparaphrase_latent_space_with_digitcount.csv'
    paraphrase_latent_space_without_digitcount_file = r'/paraphrase_latent_space_without_digitcount.csv'
    non_paraphrase_latent_space_without_digitcount_file = r'/nonparaphrase_latent_space_without_digitcount.csv'
    sharedParameters = latentspace.SharedParameters(fastText_file, paraphrase_latent_space_with_digitcount_file, non_paraphrase_latent_space_with_digitcount_file, paraphrase_latent_space_without_digitcount_file, non_paraphrase_latent_space_without_digitcount_file)


    ###initialize default settings
    device = "/device:gpu:0"
    defaultSettings = latentspace.DefaultSettings(device, sharedParameters)

    ###task dataset loading
    SICKE_file = open(r"/SICK.txt", 'r', encoding='utf-16')
    SICKE_list = SICKE_file.readlines()
    SICKE_file.close()


    ###get sentence pairs from task dataset
    ###we use the hyperparameters of BERT fine-tuning like learning-rate 5e-5, so dev set is combined with training set for training
    train_sentence_pair_list = []
    train_label_list = []
    eval_label_list = []

    for record in SICKE_list:
        record = record.rstrip("\n")
        all_values = record.split('\t')
        if all_values[11] == "SemEval_set":
            continue
        sentence1 = all_values[1]
        sentence2 = all_values[2]
        label0 = float(all_values[4]) / 5.0
        label1 = 1.0 - label0
        if all_values[11] == "TEST":
            continue
        else:
            train_sentence_pair_list.append((sentence1, sentence2))
            train_label_list.append([label0, label1])
            eval_label_list.append(label0)


    ###get features for sentence pairs
    train_feature_list_default_settings_without_digitcount = []
    for sent1, sent2 in train_sentence_pair_list:
        latent_s = defaultSettings.get_features_for_NLI_STS_tasks(sent1, sent2, with_digit_count=False)
        train_feature_list_default_settings_without_digitcount.append(latent_s)


    ###initialize dataloaders
    train_inputs = torch.tensor(train_feature_list_default_settings_without_digitcount).float()
    train_labels = torch.tensor(train_label_list)
    train_dataset = TensorDataset(train_inputs, train_labels)
    train_dataloader_default_settings_without_digitcount = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=8
    )


    ###start training
    criterion = torch.nn.MSELoss()
    device = torch.device('cuda')

    ###train train_dataloader_default_settings_without_digitcount
    classifier = Classifier()
    classifier.cuda()

    epochs = 100
    optimizer = AdamW(classifier.parameters(), lr=5e-5, eps=1e-8)
    total_steps = len(train_dataloader_default_settings_without_digitcount) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_p = 0
    best_s = 0
    total_t0 = time.time()

    for epoch_i in range(0, epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        classifier.train()

        for step, batch in enumerate(train_dataloader_default_settings_without_digitcount):

            b_inputs = batch[0].to(device)
            b_labels = batch[1].to(device)

            classifier.zero_grad()
            optimizer.zero_grad()

            outputs = classifier(b_inputs)
            loss = criterion(outputs, b_labels)

            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader_default_settings_without_digitcount), elapsed))
                avg_train_loss = total_train_loss / step
                print("  40-step training loss: {0:.2f}".format(avg_train_loss))

        avg_train_loss = total_train_loss / len(train_dataloader_default_settings_without_digitcount)

        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        print("")
        print("Running Validation...")

        t0 = time.time()

        classifier.eval()

        true_preds = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        total_predict_labels = []

        for latent_s in train_feature_list_default_settings_without_digitcount:
            score_dist = classifier(torch.FloatTensor([latent_s]).to(device)).detach().cpu().numpy().tolist()[0]
            score = score_dist[0]

            total_predict_labels.append(score)

        pearsonr = stats.pearsonr(total_predict_labels, eval_label_list)[0]
        print("  pearsonr: {0:.3f}".format(pearsonr))
        spearmanr = stats.spearmanr(total_predict_labels, eval_label_list)[0]
        print("  spearmanr: {0:.3f}".format(spearmanr))

        if pearsonr >= best_p:
            best_p = pearsonr
            best_s = spearmanr
            torch.save(classifier, r"/classifier.pth")

    print("")
    print("This dataloader training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    print("best pearsonr among all epochs is: " + str(best_p))

    print("best spearmanr among all epochs is: " + str(best_s))
