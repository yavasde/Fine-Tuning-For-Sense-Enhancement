from transformers import BertTokenizer
import torch
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score
from source.models import WordOccurrenceClassifier
import source.utils


def fine_tuning_task(learning_rate=None, num_epochs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WordOccurrenceClassifier().to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    source.utils.freeze_model_layers(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = BCEWithLogitsLoss()

    train_dataloader, validation_dataloader = source.utils.prepare_data_for_training(task_adaptation=True)

    best_loss = 100
    for epoch in range(num_epochs):
        train_running_loss = 0
        for batch_id, (data, labels, indeces) in enumerate(train_dataloader):
            labels = labels.to(device=device)

            outputs = model(data, indeces, device)
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
            f"Average Loss: {avg_loss_train}"
        )
        
        val_running_loss = 0
        all_predictions_val, all_labels_val = [], []
        with torch.no_grad():
            for data, labels, indeces in validation_dataloader:
                labels = labels.to(device=device)

                outputs = model(data, indeces, device)
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
            torch.save(model, f'trained_models/task_adapted_BERT.pth')
            best_loss = val_avg_loss
        else:
            print(f"Best Model is trained for {epoch}")
            break
    

