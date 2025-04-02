from transformers import BertTokenizer
import torch
import torch.nn as nn
import source.utils
from source.models import BERTforSPL


class CrossEntropyWithTemperature(nn.Module):
    def __init__(self, temperature=1.0):
        super(CrossEntropyWithTemperature, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits, labels):
        scaled_logits = logits / self.temperature
        return self.criterion(scaled_logits, labels)


def fine_tuning_SPL(learning_rate=None, num_epochs=None, tau=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTforSPL().to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    source.utils.freeze_model_layers(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyWithTemperature(temperature=tau)

    train_dataloader, validation_dataloader = source.utils.prepare_data_for_training()

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
                val_n_correct += source.utils.calculate_accuracy(labels[mask], predictions[mask])
                val_n_samples += mask.sum().item()

            val_avg_loss = val_running_loss / len(validation_dataloader)
            val_accuracy = 100.0 * val_n_correct / val_n_samples
            print(
                f"Validation:\tEpoch: {epoch+1},\tAccuracy: {val_accuracy},\t"
                f"Loss: {val_avg_loss}"
            )

        if val_avg_loss <= best_loss:
            torch.save(model, f"trained_models/SPL-{tau}.pth")
            best_loss = val_avg_loss
        else:
            print(f"Best Model is trained for {epoch}")
            break
