import numpy as np
from sklearn.decomposition import PCA
import torch
import copy


def turn_to_dataset(isotropic_train_data, isotropic_dev_data, isotropic_test_data, wic_dataset):
    final_isotropic_train_data = []
    for i in range(0, len(isotropic_train_data), 2):
      combined_emb = np.concatenate((isotropic_train_data[i], isotropic_train_data[i+1]), axis=0)
      combined_emb = torch.tensor(combined_emb)
      combined_emb = combined_emb.unsqueeze(0)
      final_isotropic_train_data.append(combined_emb)

    final_isotropic_dev_data = []
    for i in range(0, len(isotropic_dev_data), 2):
      combined_emb = np.concatenate((isotropic_dev_data[i], isotropic_dev_data[i+1]), axis=0)
      combined_emb = torch.tensor(combined_emb)
      combined_emb = combined_emb.unsqueeze(0)
      final_isotropic_dev_data.append(combined_emb)

    final_isotropic_test_data = []
    for i in range(0, len(isotropic_test_data), 2):
      combined_emb = np.concatenate((isotropic_test_data[i], isotropic_test_data[i+1]), axis=0)
      combined_emb = torch.tensor(combined_emb)
      combined_emb = combined_emb.unsqueeze(0)
      final_isotropic_test_data.append(combined_emb)

    new_wic_dataset = {"train": {"inputs": final_isotropic_train_data, "labels": wic_dataset["train"]["labels"]}, 
                      "dev": {"inputs": final_isotropic_dev_data, "labels": wic_dataset["dev"]["labels"]},
                      "test": {"inputs": final_isotropic_test_data, "labels": wic_dataset["test"]["labels"]}}
    
    return new_wic_dataset

def isotropize_embeddings(word_vectors, pca_dim):
    mean_vector = np.mean(word_vectors, axis=0)
    centered_vectors = [vector - mean_vector for vector in word_vectors]
    pca = PCA()
    pca.fit(np.array(centered_vectors))

    processed_vectors = []
    for vector in centered_vectors:
        projections = sum((vector.dot(pc) * pc) for pc in pca.components_[:pca_dim])
        processed_vector = vector - projections
        processed_vectors.append(processed_vector)

    return processed_vectors

def isotropize_wic_dataset(wic_dataset, pca_dim):
  all_embeddings = []
  for i in range(len(wic_dataset["train"]["inputs"])):
      w1_emb = torch.tensor_split(wic_dataset["train"]["inputs"][i], 2, dim=1)[0]
      w2_emb = torch.tensor_split(wic_dataset["train"]["inputs"][i], 2, dim=1)[1]
      all_embeddings.append(w1_emb)
      all_embeddings.append(w2_emb)
  for i in range(len(wic_dataset["dev"]["inputs"])):
      w1_emb = torch.tensor_split(wic_dataset["dev"]["inputs"][i], 2, dim=1)[0]
      w2_emb = torch.tensor_split(wic_dataset["dev"]["inputs"][i], 2, dim=1)[1]
      all_embeddings.append(w1_emb)
      all_embeddings.append(w2_emb)
  for i in range(len(wic_dataset["test"]["inputs"])):
    w1_emb = torch.tensor_split(wic_dataset["test"]["inputs"][i], 2, dim=1)[0]
    w2_emb = torch.tensor_split(wic_dataset["test"]["inputs"][i], 2, dim=1)[1]
    all_embeddings.append(w1_emb)
    all_embeddings.append(w2_emb)

  all_embeddings = np.asarray(all_embeddings)
  all_embeddings = np.squeeze(all_embeddings, axis=1)

  len_train = len(wic_dataset["train"]["inputs"]) * 2
  len_dev = len(wic_dataset["dev"]["inputs"]) * 2
  len_test = len(wic_dataset["test"]["inputs"]) * 2

  isotropic_representations = isotropize_embeddings(copy.deepcopy(all_embeddings), pca_dim)
  isotropic_train_data = isotropic_representations[:len_train]
  isotropic_dev_data = isotropic_representations[len_train:len_train+len_dev]
  isotropic_test_data = isotropic_representations[len_train+len_dev:len_train+len_dev+len_test]

  isotropic_dataset = turn_to_dataset(isotropic_train_data, isotropic_dev_data, isotropic_test_data, wic_dataset)
  return isotropic_dataset

def istoropize_datasets(topology_data, wic_dataset):
    topology_vectors = np.asarray(topology_data["inputs"])

    pca_dim = 1
    print(f"Isotopization with Dimension {pca_dim}")
    wic_dataset_copy = copy.deepcopy(wic_dataset)
    topology_vectors_copy = copy.deepcopy(topology_vectors)

    isotropized_topology_vectors = isotropize_embeddings(topology_vectors_copy, pca_dim)
    isotropized_topology_dataset  = {"inputs": [torch.tensor(v) for v in isotropized_topology_vectors], "labels": topology_data["labels"]}

    isotropized_wic_dataset = isotropize_wic_dataset(wic_dataset_copy, pca_dim)

    return isotropized_topology_dataset, isotropized_wic_dataset

