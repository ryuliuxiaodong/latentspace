### Here is the pseudo code about how to use our encapsulated programming interface.
### Concrete examples of real code are under the directory of "classifier-tuning example"
### latentspace.py is available under the directory of "encapsulated programming interface"
### Our pre-trained latent spaces use the format of ".csv" file, which is vulnerable. So please send email to "ryuliuxiaodong@gmail.com" if you need them for your application


## import python package
import latentspace

### initialize SharedParameters: pre-trained fastText and latent spaces
fastText_file = "directory/crawl-300d-2M-subword.vec"
paraphrase_latent_space_with_digitcount_file = r'directory/paraphrase_latent_space_with_digitcount.csv'
non_paraphrase_latent_space_with_digitcount_file = r'directory/non_paraphrase_latent_space_with_digitcount.csv'
paraphrase_latent_space_without_digitcount_file = r'directory/paraphrase_latent_space_without_digitcount.csv'
non_paraphrase_latent_space_without_digitcount_file = r'directory/non_paraphrase_latent_space_without_digitcount.csv'
sharedParameters = latentspace.SharedParameters(fastText_file, 
                                                paraphrase_latent_space_with_digitcount_file,
                                                non_paraphrase_latent_space_with_digitcount_file,
                                                paraphrase_latent_space_without_digitcount_file,
                                                non_paraphrase_latent_space_without_digitcount_file)
                                                    
                                                    
### computational resource
device = "/device:gpu:0"


### initialize default settings
defaultSettings = latentspace.DefaultSettings(device, sharedParameters)


### MRPC task
### get features for sentence pairs
training_feature_list = []
training_label_list = []
for sentence1, sentence2 in MRPC_Trainig_Dataset:
    feature = defaultSettings.get_features_for_PI_task(sentence1, sentence2, with_digit_count=False)
    training_feature_list.append(feature)
    training_label_list.append(MRPC_Trainig_Dataset.label)
### Feed faetures and labels to dataloader
MRPC_Training_dataLoader = DataLoader(training_feature_list, training_label_list)

### Twitter-URL task
### detect occurence difference
detector = latentspace.CharacteristicsDetector(Twitter-URL_sentence_pairs_list, label_list)
detector.detect(histogram_bins=100)
### initialize adjusted settings
adjustedSettings = latentspace.AdjustedSettings(device, sharedParameters)
adjustedSettings.adjust_settings(senlendiff_threshold=15, senlen_threshold=18, senlen_reverse=True, jdist_threshold=0.86)
### get features for sentence pairs
training_feature_list = []
training_label_list = []
for sentence1, sentence2 in Twitter-URL_Trainig_Dataset:
    feature = adjustedSettings.get_features_for_PI_task(sentence1, sentence2)
    training_feature_list.append(feature)
    training_label_list.append(Twitter-URL_Trainig_Dataset.label)
### Feed faetures and labels to dataloader
Twitter-URL_Training_dataLoader = DataLoader(training_feature_list, training_label_list)

### SICK-E task
### get features for sentence pairs
training_feature_list = []
training_label_list = []
for sentence1, sentence2 in SICKE_Trainig_Dataset:
    feature = defaultSettings.get_features_for_NLI_STS_tasks(sentence1, sentence2)
    training_feature_list.append(feature)
    training_label_list.append(SICKE_Trainig_Dataset.label)
### Feed faetures and labels to dataloader
SICKE_Training_dataLoader = DataLoader(training_feature_list, training_label_list)

### Example of how to calculate cosine similarity inparaphrase/non-paraphrase latent spaces
### with digit-count scheme
S1_p_latent_representation, S2_p_latent_representation = defaultSettings.get_paraphrase_latent_representations_for_sentence_pair(S1, S2)
cosine_similarity(S1_p_latent_representation, S2_p_latent_representation)
S1_np_latent_representation, S2_np_latent_representation = defaultSettings.get_nonparaphrase_latent_representations_for_sentence_pair(S1, S2)
cosine_similarity(S1_np_latent_representation, S2_np_latent_representation)

### without digit-count scheme
S1_p_latent_representation, S2_p_latent_representation = defaultSettings.get_paraphrase_latent_representations_for_sentence_pair(S1, S2, with_digit_count=False)
cosine_similarity(S1_p_latent_representation, S2_p_latent_representation)
S1_np_latent_representation, S2_np_latent_representation = defaultSettings.get_nonparaphrase_latent_representations_for_sentence_pair(S1, S2, with_digit_count=False)
cosine_similarity(S1_np_latent_representation, S2_np_latent_representation)
