import pickle
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import tarfile


def freeze_model_layers(model, num_layers_to_freeze=8):
    if num_layers_to_freeze == 1:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
    else:
        for layer in model.bert.encoder.layer[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.x_data = inputs
        self.y_data = labels
        self.n_samples = len(self.x_data)

    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i]
    
    def __len__(self):
        return self.n_samples
    
class CustomDataset_task(Dataset):
    def __init__(self, inputs, labels, indeces):
        self.x_data = inputs
        self.y_data = labels
        self.indeces = indeces
        self.n_samples = len(self.x_data)

    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i], self.indeces[i]
    
    def __len__(self):
        return self.n_samples

def custom_collate_fn(batch):
    input_ids = [item[0]['input_ids'] for item in batch]
    token_type_ids = [item[0]['token_type_ids'] for item in batch]
    attention_mask = [item[0]['attention_mask'] for item in batch]

    data = {"input_ids": pad_sequence([t.squeeze(0) for t in input_ids], batch_first=True, padding_value=0),
            "token_type_ids": pad_sequence([t.squeeze(0) for t in token_type_ids], batch_first=True, padding_value=0),
            "attention_mask": pad_sequence([t.squeeze(0) for t in attention_mask], batch_first=True, padding_value=0)}
    labels = torch.tensor([item[1] for item in batch]) 
    indices = [item[2] for item in batch] 
    return data, labels, indices
    
def prepare_data_for_training(batch_size=8, task_adaptation=False):
    if task_adaptation:
        with tarfile.open("data/semcor_binary.tar.gz", "r:gz") as tar:
            tar.extractall("data/semcor_data")

        with open("data/semcor_data/data/semcor_binary.pickle", "rb") as data_file:
            dataset = pickle.load(data_file)
            X_train, X_rest, y_train, y_rest, z_train, z_rest = train_test_split(
                                                            dataset["sentences"],
                                                            dataset["labels"],
                                                            dataset["indeces"],
                                                            test_size=0.3,
                                                            random_state=42)
            X_test, X_val, y_test, y_val, z_test, z_val = train_test_split(X_rest,
                                                                           y_rest,
                                                                           z_rest,
                                                                           test_size=0.5,
                                                                           random_state=42)

            train_data = {"sentences": X_train, "labels": y_train, "indeces": z_train}
            val_data = {"sentences": X_val, "labels": y_val, "indeces": z_val}

            train_dataset = CustomDataset_task(train_data["sentences"], train_data["labels"], train_data["indeces"])
            val_dataset = CustomDataset_task(val_data["sentences"], val_data["labels"], val_data["indeces"])
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
            validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    else:
        with tarfile.open("data/semcor.tar.gz", "r:gz") as tar:
            tar.extractall("data/semcor_data")

        with open("data/semcor_data/data/semcor.pickle", "rb") as data_file:
            dataset = pickle.load(data_file)   

        X_train, y_train, X_val, y_val, X_test, y_test = split_data(dataset)

        train_data = {"inputs": X_train, "labels": y_train}
        val_data = {"inputs": X_val, "labels": y_val}

        train_dataset = CustomDataset(train_data["inputs"], train_data["labels"])
        val_dataset = CustomDataset(val_data["inputs"], val_data["labels"])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        validation_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, validation_dataloader

def calculate_accuracy(masked_labels, masked_predictions):
    correct = (masked_labels == masked_predictions).sum().item()
    return correct

def split_data(dataset):
    X_train, X_rest, y_train, y_rest = train_test_split(dataset["inputs"],
                                                        dataset["labels"],
                                                        test_size=0.3,
                                                        random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test
