import torch
from scipy.spatial.distance import cosine
import numpy as np


def calculate_accuracy(labels, predictions):
    correct_predictions = sum(1 for label, prediction in zip(labels, predictions) if label == prediction)
    accuracy = correct_predictions / len(labels)
    return accuracy

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1.squeeze(), vec2.squeeze())
    
def calculate_word_similarity(dataset):
    labels = []
    similarities = []
    for i in range(len(dataset["inputs"])):
        w1_emb = torch.tensor_split(dataset["inputs"][i], 2, dim=1)[0]
        w2_emb = torch.tensor_split(dataset["inputs"][i], 2, dim=1)[1]
        if w1_emb is not None and w2_emb is not None:
            label = dataset['labels'][i]
            similarity = cosine_similarity(w1_emb, w2_emb)
            labels.append(label)
            similarities.append(similarity)
    return similarities, labels

def eval_for_threshold(results_dict, similarities, labels):
    for threshold in results_dict.keys():
        predictions = []
        for i in range(len(similarities)):
            if similarities[i] >= threshold:
                predictions.append(1.)
            else:
                predictions.append(0.)
        results_dict[threshold] = calculate_accuracy(labels, predictions)
    return results_dict

def select_best_threshold(wic_dataset):
    thresholds = np.arange(0.0, 1.0, 0.02)
    threshold_results = {"dev": {t: [] for t in thresholds}, 
                    "test": {t: [] for t in thresholds}}

    dev_similarities, dev_labels = calculate_word_similarity(wic_dataset['dev'])
    test_similarities, test_labels = calculate_word_similarity(wic_dataset['test'])

    eval_for_threshold(threshold_results['dev'], dev_similarities, dev_labels)
    eval_for_threshold(threshold_results['test'], test_similarities, test_labels)

    best_threshold_result = 0.2
    for threshold_result in threshold_results['dev'].items():
        if threshold_result[1] > best_threshold_result:
            best_threshold_result = threshold_result[1]
            best_threshold = threshold_result[0]

    return threshold_results['test'][best_threshold]
