import argparse
from evaluation.eval_data_preparation import prepare_topology_dataset, prepare_wic_dataset, load_models
from evaluation.topology import evaluate_topology
from evaluation.wic_threshold_test import select_best_threshold
from pathlib import Path
from evaluation.isotropy_improvement import istoropize_datasets
import pickle

"""
This script evaluated the embedding space of a model in terms of topology and task performance.
Furthermore, it applies isotropization post-processing to the embeddings if specified before evaluation.

Usage:
    python eval_main.py model_type tau --isotropized

Arguments:
- model_type (str): Type of the model (BERT or fine-tuned for SCL, SPL or TASK)
- tau (float) (optional): Temperature of the loss (only for SCL and SPL)
- isotropized (boolean) (optional): Apply isotropization

Example:
    python eval_main.py BERT --isotropized
"""


def main():
    parser = argparse.ArgumentParser(
        description="
This script evaluated the embedding space of a model in terms of topology and task performance.
Furthermore, it applies isotropization post-processing to the embeddings if specified before evaluation.
"
    )
    parser.add_argument("model_type", type=str, help="Type of the model (BERT or fine-tuned for SCL, SPL or TASK)")
    parser.add_argument(
        "tau",
        type=float,
        help="Temperature of the loss (only for SCL and SPL)",
    )
    parser.add_argument('--isotropized', dest='isotropized', action='store_true',
                    help='Apply isotropization.')


    args = parser.parse_args()
    model_type = args.model_type
    tau = args.tau

    print(f"Model Name: {model_type}-{tau}")
    print("Model loading\n")

    model, tokenizer = load_models(model_type, tau)

    # Prepare Evaluation Data
    print("Preparing WiC data\n")
    wic_dataset_path = f"evaluation/eval_datasets/wic/wic_dataset_{model_type}-{tau}.pickle"
    wic_dataset_file = Path(wic_dataset_path)
    if not wic_dataset_file.is_file():
        wic_dataset = prepare_wic_dataset(model, tokenizer, model_type=model_type, dataset_path=wic_dataset_path)
    else:
        with open(wic_dataset_path, "rb") as wic_file:
            wic_dataset = pickle.load(wic_file)

    print("Preparing Topology data\n")
    topology_dataset_path = f"evaluation/eval_datasets/topology/topology_dataset_{model_type}-{tau}.pickle"
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
