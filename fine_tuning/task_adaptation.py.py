from transformers import BertTokenizer
import pickle
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch
import wandb    
import argparse
import torch.nn as nn
from transformers import BertModel
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
from models import WordOccurrenceClassifier
import utils


def custom_collate_fn(batch):
    input_ids = [item[0]['input_ids'] for item in batch]
    token_type_ids = [item[0]['token_type_ids'] for item in batch]
    attention_mask = [item[0]['attention_mask'] for item in batch]

    data = {"input_ids": pad_sequence([t.squeeze(0) for t in input_ids], batch_first=True, padding_value=0),
            "token_type_ids": pad_sequence([t.squeeze(0) for t in token_type_ids], batch_first=True, padding_value=0),
            "attention_mask": pad_sequence([t.squeeze(0) for t in attention_mask], batch_first=True, padding_value=0)}
    labels = torch.tensor([item[1] for item in batch]) 
    indices = [item[2] for item in batch]  # Keeps indices as a list
    return data, labels, indices


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('lr', type=str, help='Learning Rate')
parser.add_argument('epoch', type=str, help='Number of Epochs')
args = parser.parse_args()

freeze_layer_no = 8
drop_out = 0.7
num_labels = 2   

with open(f"fine_tuning/data/semcor_binary.pickle", "rb") as data_file:
    dataset = pickle.load(data_file)

model = WordOccurrenceClassifier(drop_out=drop_out).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
utils.freeze_model_layers(model, freeze_layer_no)

batch_size = 8
learning_rate = float(args.lr)
num_epochs = int(args.epoch)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = BCEWithLogitsLoss()

X_train, X_rest, y_train, y_rest, z_train, z_rest = train_test_split(dataset["sentences"], dataset["labels"], dataset["indeces"], test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val, z_test, z_val = train_test_split(X_rest, y_rest, z_rest, test_size=0.5, random_state=42)

train_data = {"sentences": X_train, "labels": y_train, "indeces": z_train}
test_data = {"sentences": X_test, "labels": y_test, "indeces": z_test}
val_data = {"sentences": X_val, "labels": y_val, "indeces": z_val}

train_dataset = utils.CustomDataset(train_data["sentences"], train_data["labels"], train_data["indeces"])
test_dataset = utils.CustomDataset(test_data["sentences"], test_data["labels"], test_data["indeces"])
val_dataset = utils.CustomDataset(val_data["sentences"], val_data["labels"], val_data["indeces"])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

best_loss = 100
for epoch in range(num_epochs):
    train_running_loss = 0
    for batch_id, (data, labels, indeces) in enumerate(train_dataloader):
        labels = labels.to(device=device)

        outputs = model(data, indeces)
        loss = criterion(outputs.squeeze(), labels.float())
        
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
    all_predictions_val, all_labels_val = [], []
    with torch.no_grad():
        for data, labels, indeces in validation_dataloader:
            labels = labels.to(device=device)
            data = data.to(device=device)
            indeces1 = indeces[0].to(device=device)
            indeces2 = indeces[1].to(device=device)

            outputs = model(data, indeces1, indeces2).to(device=device)
            loss = criterion(outputs.squeeze(), labels.float())
            val_running_loss += loss.item()

            all_predictions_val.extend((outputs.squeeze() > 0.5).cpu().numpy())
            all_labels_val.extend(labels.cpu().numpy())

        val_avg_loss = val_running_loss / len(validation_dataloader)
        val_accuracy = accuracy_score(all_labels_val, all_predictions_val)
        print(
            f"Validation:\tEpoch: {epoch+1},\tAccuracy: {val_accuracy},\t"
            f"Loss: {val_avg_loss}"
        )
        

    if val_avg_loss <= best_loss:
        torch.save(model, f'models/task_adapted_BERT.pth')
        best_loss = val_avg_loss
    

