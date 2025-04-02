import argparse
from evaluation.eval_data_preparation import prepare_topology_dataset, prepare_wic_dataset, load_models
from evaluation.topology import evaluate_topology
from evaluation.wic_threshold_test import select_best_threshold
from source.isotropy_improvement import istoropize_datasets

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
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=str, help="Type of the model"
                                                            "(BERT or fine-tuned for SCL, SPL or TASK)")
    parser.add_argument("-tau", type=float, help="Optional argument. Temperature of the loss"
                                                        " (only for SCL and SPL)")
    parser.add_argument('--isotropized', action='store_true', help="Optional argument."
                                                                            "Apply isotropization.")

    args = parser.parse_args()
    model_type = args.model_type
    tau = args.tau
    if tau and model_type in ["BERT", "TASK"]:
        print(f"Tau is not required for model type {model_type}."
              " Evaluation will continue without considering tau.")
    
    if not tau and model_type in ["SCL", "SPL"]:
        raise Exception(f"Tau is required for model type {model_type}.")

    print(f"Model Name: {model_type}-{tau}")
    print("Model loading\n")

    model, tokenizer = load_models(model_type, model_name=tau)

    # Prepare Evaluation Data
    print("Preparing WiC data\n")
    wic_dataset = prepare_wic_dataset(model, tokenizer, model_type=model_type)

    print("Preparing Topology data\n")
    topology_dataset = prepare_topology_dataset(model, tokenizer, model_type=model_type)

    if args.isotropized:
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
