from transformers import AutoTokenizer, AutoModel
import torch
from tqdm.auto import tqdm
import pickle
from source.utils import split_data
import tarfile
from pathlib import Path


def extract_all_noun_embeddings(model, tokenizer, sentence, labels, remove=[0, -100], model_type=''):
    noun_embeddings = []
    noun_labels = []
    nouns = []
    tokenized_sentence = tokenizer(sentence, truncation=True, return_tensors="pt")
    if model_type in ["SPL", "SCL", "TASK"]:
        output = model.bert(
            **tokenized_sentence, output_hidden_states=True
        ).hidden_states
    else:
        output = model(
            **tokenized_sentence, output_hidden_states=True
        ).hidden_states
    for i in range(len(tokenized_sentence["input_ids"][0])):
        if labels[i] not in remove:
            noun_embedding = output[-1][:, i, :][0].clone().detach()
            noun_embeddings.append(noun_embedding)
            noun_labels.append(labels[i].item())
            nouns.append(
                tokenizer.convert_ids_to_tokens(
                    tokenized_sentence["input_ids"][0][i].item()
                )
            )
    return noun_embeddings, noun_labels, nouns

def extract_target_word_embedding(model, tokenizer, sentence, word_idx, model_type=''):
    tokenized_sentence = tokenizer(sentence, truncation=True, return_tensors="pt")
    if model_type in ["SPL", "SCL", "TASK"]:
        output = model.bert(**tokenized_sentence, output_hidden_states=True).hidden_states
    else:
        output = model(**tokenized_sentence, output_hidden_states=True).hidden_states
    if tokenized_sentence.word_ids().count(int(word_idx)) > 1:
        embeddings = []
        for i in range(len(tokenized_sentence.word_ids())):
            if tokenized_sentence.word_ids()[i] == int(word_idx):
                token_vector = [output[layer_no][:, i, :].detach() for layer_no in range(9,13)]
                avg_token_vector = torch.mean(torch.stack(token_vector), dim=0)
                embeddings.append(avg_token_vector.clone().detach())
        word_embedding = torch.mean(torch.stack(embeddings), dim=0)
    else:
        token_idx = tokenized_sentence.word_ids().index(int(word_idx))
        token_vector = [output[layer_no][:, token_idx, :].detach() for layer_no in range(9,13)]
        word_embedding = torch.mean(torch.stack(token_vector), dim=0)
    return word_embedding

def extract_wic_data(model, tokenizer, model_type='', data_type=''):
    print(f"WiC {data_type} data\n")
    data = {"inputs": [], "labels": []}
    if data_type == 'train':
        progress_bar_len = 5428
    elif data_type == 'dev':
        progress_bar_len = 638
    elif data_type == 'test':
        progress_bar_len = 1400

    progress_bar = tqdm(range(progress_bar_len))
    with open(f"data/wic/{data_type}/{data_type}.data.txt", 'r', encoding='utf8') as train_data_file:
        for line in train_data_file.readlines():
            sentence1 = line[:-1].split("\t")[3]
            sentence2 = line[:-1].split("\t")[4]
            indeces = line[:-1].split("\t")[2]
            word1_idx = indeces.split("-")[0]
            word2_idx = indeces.split("-")[1]
            word1_vector = extract_target_word_embedding(model,
                                                         tokenizer,
                                                         sentence1,
                                                         word1_idx,
                                                         model_type=model_type)
            word2_vector = extract_target_word_embedding(model,
                                                         tokenizer,
                                                         sentence2,
                                                         word2_idx,
                                                         model_type=model_type)
            data['inputs'].append(torch.cat((word1_vector, word2_vector), 1))
            progress_bar.update(1)    

    with open(f"data/wic/{data_type}/{data_type}.gold.txt", 'r', encoding='utf8') as train_data_file:
        for line in train_data_file.readlines():
            if line[:-1] == "F":
                label = 0.
            elif line[:-1] == "T":
                label = 1.
            data['labels'].append(torch.tensor(label))
    return data 

def load_models(model_type, model_name=None):
    bert_model_name = "bert-base-uncased"
    if model_type == "BERT":
        model_name = bert_model_name
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        model = AutoModel.from_pretrained(bert_model_name)
    elif model_type == "TASK":
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        model = torch.load(f'trained_models/{model_type}.pth', map_location=torch.device('cpu'), weights_only=False)
        model.eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        model = torch.load(f'trained_models/{model_type}-{model_name}.pth', map_location=torch.device('cpu'), weights_only=False)
        model.eval()
    return model, tokenizer

def prepare_wic_dataset(model, tokenizer, model_type=''):
    test_data = extract_wic_data(model, tokenizer, model_type=model_type, data_type="test")
    dev_data = extract_wic_data(model, tokenizer, model_type=model_type, data_type="dev")
    train_data = extract_wic_data(model, tokenizer, model_type=model_type, data_type="train")

    wic_dataset = {"train": train_data, "test": test_data, "dev": dev_data}
    return wic_dataset

def prepare_topology_dataset(model, tokenizer, model_type=''):
    semcor_data_file = "data/semcor_data/data/semcor.pickle"
    my_file = Path(semcor_data_file)
    if not my_file.is_file():
        with tarfile.open("data/semcor.tar.gz", "r:gz") as tar:
            tar.extractall("data/semcor_data")

    with open(semcor_data_file, "rb") as data_file:
        semcor_data = pickle.load(data_file)   

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(semcor_data)

    noun_embeddings = []
    noun_labels = []
    all_nouns = []

    progress_bar = tqdm(range(len(X_test)))
    for i in range(len(X_test)):
        embeddings, labels, nouns = extract_all_noun_embeddings(model, tokenizer, X_test[i], y_test[i], model_type=model_type)
        noun_embeddings += embeddings
        noun_labels += labels
        all_nouns += nouns
        progress_bar.update(1)

    final_nouns = []
    final_embeddings = []
    final_labels = []
    index_count = 0
    while index_count in range(len(noun_embeddings) - 1):
        subword_embeddings = [noun_embeddings[index_count]]
        first_word_label = noun_labels[index_count]
        subwords = [all_nouns[index_count]]
        next_label_id = index_count + 1
        next_label = noun_labels[next_label_id]
        next_subword = all_nouns[next_label_id].replace("#", "")
        if first_word_label % 2 != 0:
            while next_label == first_word_label + 1:
                subword_embeddings.append(noun_embeddings[next_label_id])
                subwords.append(next_subword)
                next_label_id += 1
                index_count += 1
                if next_label_id < len(noun_embeddings):
                    next_label = noun_labels[next_label_id]
                    next_subword = all_nouns[next_label_id].replace("#", "")
                else:
                    next_label = 0
            final_embeddings.append(torch.mean(torch.stack(subword_embeddings), dim=0))
            final_labels.append(first_word_label)
            final_nouns.append("".join(subwords))
            index_count += 1

    test_data = {"inputs": final_embeddings, "labels": final_labels, "nouns": final_nouns}
    return test_data
