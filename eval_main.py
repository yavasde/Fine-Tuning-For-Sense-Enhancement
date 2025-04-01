import argparse
from evaluation.eval_data_preparation import prepare_topology_dataset, prepare_wic_dataset, load_models
from evaluation.topology import evaluate_topology
from evaluation.wic_threshold_test import select_best_threshold
from pathlib import Path
from evaluation.isotropy_improvement import istoropize_datasets
import pickle

"""
This script automatically annotates the clusters of a word based on the sentence 
features responsible for the formation of the clusters. It performs the following steps:

1. Extracts sentences containing the target word and its alternative forms.
2. Annotates these sentences with linguistic features (sentence context features).
3. Extracts embeddings for the target word in each sentence using a pre-trained language model.
4. Clusters the embeddings.
5. Selects the top features with recursive feature elimination.
6. Annotate the clusters with informative features and writes the results to files 
and plots the clusters with annotated features.

Usage:
    python main.py target_word alternative_forms n_sentences n_features

Arguments:
- target_word (str): Target word to analyze.
- alternative_forms (str): Alternative forms of the word, e.g., 
    "buy\bought" for "buy".
- n_sentences (int): Number of sentences to extract.
- n_features (int): Number of sentence features to select.

Example:
    python script.py buy buy\bought 200 50
"""


def main():
    parser = argparse.ArgumentParser(
        description="Automatically annotates the clusters of a word based on the"
        "sentence features resposible for the formation of the clusters."
    )
    parser.add_argument("model_type", type=str, help="Target word")
    parser.add_argument(
        "model_name",
        type=str,
        help="Alternative forms of the word like 'buy' and 'bought' for 'buy'."
        "Give as: buy\\bought",
    )
    parser.add_argument('--isotropized', dest='isotropized', action='store_true',
                    help='Set the flag value to True.')


    args = parser.parse_args()
    model_type = args.model_type
    model_name = args.model_name

    print(f"Model Name: {model_type}-{model_name}")
    print("Model loading\n")

    model, tokenizer = load_models(model_type, model_name)

    # Prepare Evaluation Data
    print("Preparing WiC data\n")
    wic_dataset_path = f"evaluation/eval_datasets/wic/wic_dataset_{model_type}-{model_name}.pickle"
    wic_dataset_file = Path(wic_dataset_path)
    if not wic_dataset_file.is_file():
        wic_dataset = prepare_wic_dataset(model, tokenizer, model_type=model_type, dataset_path=wic_dataset_path)
    else:
        with open(wic_dataset_path, "rb") as wic_file:
            wic_dataset = pickle.load(wic_file)

    print("Preparing Topology data\n")
    #prepare semcor?
    topology_dataset_path = f"evaluation/eval_datasets/topology/topology_dataset_{model_type}-{model_name}.pickle"
    topology_dataset_file = Path(topology_dataset_path)
    if not topology_dataset_file.is_file():
        topology_dataset = prepare_topology_dataset(model, tokenizer, model_type=model_type, dataset_path=topology_dataset_path)
    else:
        with open(topology_dataset_path, "rb") as topology_file:
            topology_dataset = pickle.load(topology_file)

    if args.isotropized == True:
        topology_dataset, wic_dataset = istoropize_datasets(topology_dataset, wic_dataset)


    # Evaluate WiC Performance
    print("Evaluating WiC")
    threshold_test_result = select_best_threshold(wic_dataset)
    print(f"WiC Threshold: {threshold_test_result}\n")

    # Evaluate Topology
    print("Evaluating Topology")
    isotropy, uniformity, alignment = evaluate_topology(topology_dataset)
    print(f"Isotropy: {isotropy}\tUniformity: {uniformity}\tAlignment: {alignment}\n")


if __name__ == "__main__":
    main()