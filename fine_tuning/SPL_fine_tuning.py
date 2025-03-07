from transformers import BertTokenizer
import pickle
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
import argparse
import torch.nn as nn
from models import BertClassifier
import utils


class CrossEntropyWithTemperature(nn.Module):
    def __init__(self, temperature=1.0):
        super(CrossEntropyWithTemperature, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits, labels):
        scaled_logits = logits / self.temperature
        return self.criterion(scaled_logits, labels)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('lr', type=str, help='Learning Rate')
parser.add_argument('epoch', type=str, help='Number of Epochs')
parser.add_argument('tau', type=str, help='Temperature of the loss')
args = parser.parse_args()

freeze_layer_no = 8
drop_out = 0.7
num_labels = 87    

with open(f"fine_tuning/data/semcor_data.pickle", "rb") as data_file:
    dataset = pickle.load(data_file)

model = BertClassifier(num_classes=num_labels, drop_out=drop_out).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
utils.freeze_model_layers(model, freeze_layer_no)

batch_size = 8
learning_rate = float(args.lr)
num_epochs = int(args.epoch)
tau = float(args.tau)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = CrossEntropyWithTemperature(temperature=tau)

X_train, X_rest, y_train, y_rest = train_test_split(dataset["inputs"], dataset["labels"], test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

train_data = {"inputs": X_train, "labels": y_train}
test_data = {"inputs": X_test, "labels": y_test}
val_data = {"inputs": X_val, "labels": y_val}

train_dataset = utils.CustomDataset(train_data["inputs"], train_data["labels"])
test_dataset = utils.CustomDataset(test_data["inputs"], test_data["labels"])
val_dataset = utils.CustomDataset(val_data["inputs"], val_data["labels"])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
validation_dataloader = DataLoader(val_dataset, batch_size=batch_size)

best_loss = 100
for epoch in range(num_epochs):
    train_running_loss = 0
    for batch_id, (data, labels) in enumerate(train_dataloader):
        labels = labels.to(device=device)
        tokenized_inputs = tokenizer(data, truncation=True, padding='max_length', return_tensors='pt').to(device=device)
        outputs = model(**tokenized_inputs).to(device=device)

        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_running_loss += loss.item()

        if (batch_id + 1) % 50 == 0:
            print(
                f"Epoch: {epoch+1}/{num_epochs},\t"
                f"Step {batch_id+1}/{len(train_dataloader)},\t"
                f"Loss: {loss.item()}"
            )

    avg_loss_train = train_running_loss / len(train_dataloader)
    print(
        f"Epoch: {epoch+1}/{num_epochs},\t"
        f"Average Loss: {avg_loss_train.item()}"
    )

    val_running_loss = 0
    val_n_correct = 0
    val_n_samples = 0
    with torch.no_grad():
        for data, labels in validation_dataloader:
            labels = labels.to(device=device)
            tokenized_inputs = tokenizer(data, truncation=True, padding='max_length', return_tensors='pt').to(device=device)

            outputs = model(**tokenized_inputs).to(device=device)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            val_running_loss += loss.item()

            predictions = torch.argmax(outputs, dim=-1)
            mask = labels != -100
            val_n_correct += utils.calculate_accuracy(labels[mask], predictions[mask])
            val_n_samples += mask.sum().item()

        val_avg_loss = val_running_loss / len(validation_dataloader)
        val_accuracy = 100.0 * val_n_correct / val_n_samples
        print(
            f"Validation:\tEpoch: {epoch+1},\tAccuracy: {val_accuracy},\t"
            f"Loss: {val_avg_loss}"
        )

    if val_avg_loss <= best_loss:
        torch.save(model, f'models/SPL_BERT_{tau}.pth')
        best_loss = val_avg_loss
