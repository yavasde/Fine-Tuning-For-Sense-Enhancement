import argparse
from fine_tuning.SCL_fine_tuning import fine_tuning_SCL
from fine_tuning.SPL_fine_tuning import fine_tuning_SPL
from fine_tuning.task_adaptation import fine_tuning_task

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
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=str, help="Target word")
    parser.add_argument('lr', type=str, help='Learning Rate')
    parser.add_argument('epoch', type=str, help='Number of Epochs')
    parser.add_argument('tau', type=str, help='Temperature of the loss')
    args = parser.parse_args()

    model_type = args.model_type
    learning_rate = float(args.lr)
    num_epochs = int(args.epoch)
    tau = float(args.tau)

    #tau optional


    print(f"Training Model for {model_type} with learning rate {learning_rate} and temp {tau}")
    if model_type == "SCL":
        fine_tuning_SCL(learning_rate=learning_rate, num_epochs=num_epochs, tau=tau)
    elif model_type == "SPL":
        fine_tuning_SPL(learning_rate=learning_rate, num_epochs=num_epochs, tau=tau)
    elif model_type == "TASK":
        fine_tuning_task(learning_rate=learning_rate, num_epochs=num_epochs)


    # # Prepare Evaluation Data
    # print("Preparing WiC data\n")
    # wic_dataset_path = f"evaluation/eval_datasets/wic/wic_dataset_{model_type}-{model_name}.pickle"
    # wic_dataset_file = Path(wic_dataset_path)
    # if not wic_dataset_file.is_file():
    #     prepare_wic_dataset(model, tokenizer, model_type=model_type, dataset_path=wic_dataset_path)

    # print("Preparing Topology data\n")
    # #prepare semcor?
    # topology_dataset_path = f"evaluation/eval_datasets/topology/topology_dataset_{model_type}-{model_name}.pickle"
    # topology_dataset_file = Path(topology_dataset_path)
    # if not topology_dataset_file.is_file():
    #     prepare_topology_dataset(model, tokenizer, model_type=model_type, dataset_path=topology_dataset_path)


    # # Evaluate WiC Performance
    # print("Evaluating WiC")
    # threshold_test_result = select_best_threshold(model_type, model_name)
    # print(f"WiC Threshold: {threshold_test_result}\n")

    # # Evaluate Topology
    # print("Evaluating Topology")
    # isotropy, uniformity, alignment = evaluate_topology(model_type, model_name)
    # print(f"Isotropy: {isotropy}\tUniformity: {uniformity}\tAlignment: {alignment}\n")


if __name__ == "__main__":
    main()