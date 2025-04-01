import argparse
from fine_tuning.SCL_fine_tuning import fine_tuning_SCL
from fine_tuning.SPL_fine_tuning import fine_tuning_SPL
from fine_tuning.task_adaptation import fine_tuning_task

"""
This script fine-tunes the BERT model with the specified method (SCL, SPL, or for Task adaptation).

Usage:
    python train_main.py finetuning_method learning_rate epoch tau

Arguments:
- finetuning_method (str): Method of fine-tuning (SCL, SPL or TASK)
- learning_rate (float): Learning Rate
- epoch (int): Number of Epochs
- tau (float): Temperature of the loss. Required with SCL and SPL.

Example:
    python train_main.py SCL 0.0001 20 0.1
"""


def main():
    parser = argparse.ArgumentParser(description="Fine-tunes the BERT model with the specified method (SCL, SPL, or for Task adaptation).")
    parser.add_argument("finetuning_method", type=str, help="Method of fine-tuning (SCL, SPL or TASK)")
    parser.add_argument('lr', type=str, help='Learning Rate')
    parser.add_argument('epoch', type=str, help='Number of Epochs')
    parser.add_argument('tau', type=str, help='Temperature of the loss. Required with SCL and SPL.')
    args = parser.parse_args()

    model_type = args.finetuning_method
    learning_rate = float(args.lr)
    num_epochs = int(args.epoch)
    if args.tau:
        tau = float(args.tau)
    else:
        if model_type in ["SCL", "SPL"]:
            raise Exception("Enter the temperature value for the loss")

    print(f"Training Model for {model_type} with learning rate {learning_rate} and temp {tau}")
    if model_type == "SCL":
        fine_tuning_SCL(learning_rate=learning_rate, num_epochs=num_epochs, tau=tau)
    elif model_type == "SPL":
        fine_tuning_SPL(learning_rate=learning_rate, num_epochs=num_epochs, tau=tau)
    elif model_type == "TASK":
        fine_tuning_task(learning_rate=learning_rate, num_epochs=num_epochs)
    else:
        raise Exception("Enter a valid fine-tuning method: SCL, SPL, or TASK")


if __name__ == "__main__":
    main()
