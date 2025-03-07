from torch.utils.data import Dataset


def freeze_model_layers(model, num_layers_to_freeze=4):
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

def calculate_accuracy(masked_labels, masked_predictions):
    correct = (masked_labels == masked_predictions).sum().item()
    return correct
