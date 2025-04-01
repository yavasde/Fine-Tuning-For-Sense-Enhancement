import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch


def calculate_sensealignment(labels, representations):
    all_sims = []
    for class_label in np.unique(labels):
        class_vectors = [
            representations[i]
            for i in range(len(labels))
            if labels[i] == class_label
        ]
        class_vectors = l2_normalize_batch(torch.stack(class_vectors))
        similarity_matrix = cosine_similarity(class_vectors)
        mask = np.eye(similarity_matrix.shape[0], dtype=bool)
        sim = similarity_matrix[~mask]
        all_sims.append(sim)

    all_sims = np.concatenate(all_sims)
    mean_similarity = np.mean(all_sims)
    return mean_similarity

def calculate_isotropy(representations):
    eig_values, eig_vectors = torch.linalg.eig(
        torch.matmul(representations.T, representations)
    )
    eig_vectors = eig_vectors.real
    max_f = -float("inf")
    min_f = float("inf")
    for i in range(eig_vectors.shape[1]):
        f = torch.matmul(representations, eig_vectors[:, i].unsqueeze(1))
        f = torch.sum(torch.exp(f))
        min_f = min(min_f, f.item())  # Convert tensor to scalar
        max_f = max(max_f, f.item())
    isotropy = min_f / max_f
    return isotropy

def l2_normalize_batch(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    return x / norm

def calculate_uniformity(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def evaluate_topology(topology_dataset):
    embeddings_normalized = l2_normalize_batch(torch.stack(topology_dataset["inputs"]))
    return calculate_isotropy(embeddings_normalized), calculate_uniformity(embeddings_normalized), calculate_sensealignment(topology_dataset["labels"], embeddings_normalized)
