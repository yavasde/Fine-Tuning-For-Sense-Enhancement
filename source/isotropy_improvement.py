import numpy as np
from sklearn.decomposition import PCA
import torch
import copy


def split_wic_data(wic_data):
    split_embeddings = []
    for i in range(len(wic_data)):
        w1_emb = torch.tensor_split(wic_data[i], 2, dim=1)[0]
        w2_emb = torch.tensor_split(wic_data[i], 2, dim=1)[1]
        split_embeddings.append(w1_emb)
        split_embeddings.append(w2_emb)
    return split_embeddings

def combine_embeddings(split_embeddings):
    combined_embeddings = []
    for i in range(0, len(split_embeddings), 2):
        combined_emb = np.concatenate((split_embeddings[i], split_embeddings[i+1]), axis=0)
        combined_emb = torch.tensor(combined_emb)
        combined_emb = combined_emb.unsqueeze(0)
        combined_embeddings.append(combined_emb)
    return combined_embeddings

def isotropize_embeddings(word_vectors, pca_dim_global):
    mean_vector = np.mean(word_vectors, axis=0)
    centered_vectors = [vector - mean_vector for vector in word_vectors]
    pca = PCA()
    pca.fit(np.array(centered_vectors))

    processed_vectors = []
    for vector in centered_vectors:
        projections = sum((vector.dot(pc) * pc) for pc in pca.components_[:pca_dim_global])
        processed_vector = vector - projections
        processed_vectors.append(processed_vector)
    return processed_vectors

def isotropize_wic_dataset(original_wic_dataset, pca_dim):
    all_embeddings = split_wic_data(original_wic_dataset["train"]["inputs"])
    all_embeddings += split_wic_data(original_wic_dataset["dev"]["inputs"])
    all_embeddings += split_wic_data(original_wic_dataset["test"]["inputs"])

    all_embeddings = np.asarray(all_embeddings)
    all_embeddings = np.squeeze(all_embeddings, axis=1)

    len_train = len(original_wic_dataset["train"]["inputs"]) * 2
    len_dev = len(original_wic_dataset["dev"]["inputs"]) * 2
    len_test = len(original_wic_dataset["test"]["inputs"]) * 2

    isotropic_representations = isotropize_embeddings(copy.deepcopy(all_embeddings), pca_dim)
    isotropic_train_embeddings = isotropic_representations[:len_train]
    isotropic_dev_embeddings = isotropic_representations[len_train:len_train+len_dev]
    isotropic_test_embeddings = isotropic_representations[len_train+len_dev:len_train+len_dev+len_test]

    isotropic_wic_dataset = {"train": {"inputs": combine_embeddings(isotropic_train_embeddings), 
                                       "labels": original_wic_dataset["train"]["labels"]}, 
                    "dev": {"inputs": combine_embeddings(isotropic_dev_embeddings), 
                            "labels": original_wic_dataset["dev"]["labels"]},
                      "test": {"inputs": combine_embeddings(isotropic_test_embeddings), 
                               "labels": original_wic_dataset["test"]["labels"]}}
    return isotropic_wic_dataset

def istoropize_datasets(topology_data, wic_dataset):
    topology_vectors = np.asarray(topology_data["inputs"])

    pca_dim = 1
    print(f"Isotopization with Dimension {pca_dim}")
    wic_dataset_copy = copy.deepcopy(wic_dataset)
    topology_vectors_copy = copy.deepcopy(topology_vectors)

    isotropized_topology_vectors = isotropize_embeddings(topology_vectors_copy, pca_dim)
    isotropized_topology_dataset  = {"inputs": [torch.tensor(v) for v in isotropized_topology_vectors], 
                                     "labels": topology_data["labels"]}

    isotropized_wic_dataset = isotropize_wic_dataset(wic_dataset_copy, pca_dim)
    return isotropized_topology_dataset, isotropized_wic_dataset
