from transformers import BertTokenizer
import torch
import source.utils 
from source.models import BERTforSCL
import torch.nn.functional as F


def contrastive_loss(word_embeddings, labels, tau=0.1):
    loss = 0.0
    labels = labels.flatten()
    word_embeddings = word_embeddings.view(-1, 768)

    mask = labels > 0
    all_anchor_labels_subword = labels[mask]
    all_anchor_embeddings_subword = word_embeddings[mask]

    all_anchor_embeddings_word = []
    all_anchor_labels_word = []
    index_count = 0
    while index_count in range(all_anchor_embeddings_subword.shape[0]-1):
        subword_embeddings = [all_anchor_embeddings_subword[index_count]]
        first_word_label = all_anchor_labels_subword[index_count]
        next_label_id = index_count + 1
        next_label = all_anchor_labels_subword[next_label_id]
        if first_word_label % 2 != 0:
            while next_label == first_word_label + 1:
                subword_embeddings.append(all_anchor_embeddings_subword[next_label_id])
                next_label_id += 1
                index_count += 1
                if next_label_id < all_anchor_embeddings_subword.shape[0]:
                    next_label = all_anchor_labels_subword[next_label_id]
                else:
                    next_label = 0
            all_anchor_embeddings_word.append(torch.mean(torch.stack(subword_embeddings), dim=0))
            all_anchor_labels_word.append(first_word_label.item())
            index_count += 1

    for idx in range(len(all_anchor_embeddings_word)):
        anchor_label = all_anchor_labels_word[idx]
        anchor_embedding = all_anchor_embeddings_word[idx]

        if list(all_anchor_labels_word).count(anchor_label) > 1:
            all_similarities = []
            positive_similarities = []
            for i in range(len(all_anchor_embeddings_word)):
                if idx != i:
                    similarity = torch.exp(F.cosine_similarity(anchor_embedding, all_anchor_embeddings_word[i], dim=0)/ tau)
                    all_similarities.append(similarity)
                    if all_anchor_labels_word[i] == anchor_label:
                        positive_similarities.append(similarity)

            all_similarity_sum = sum(all_similarities)
            log_positive_similarities = []
            for similarity in positive_similarities:
                log_positive_similarities.append(torch.log(similarity/all_similarity_sum))
                
            pos_sim = sum(log_positive_similarities)/len(log_positive_similarities)
            loss -= pos_sim
    return loss 

def fine_tuning_SCL(learning_rate=None, num_epochs=None, tau=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTforSCL().to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    source.utils.freeze_model_layers(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader, validation_dataloader = source.utils.prepare_data_for_training()

    best_loss = 1000
    for epoch in range(num_epochs):
        train_running_loss = 0
        for batch_id, (data, labels) in enumerate(train_dataloader):
            labels = labels.to(device=device)
            tokenized_inputs = tokenizer(data, truncation=True, padding='max_length', return_tensors='pt').to(device=device)

            outputs = model(**tokenized_inputs).last_hidden_state.to(device=device)
            loss = contrastive_loss(outputs, labels, tau=tau)
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
        with torch.no_grad():
            for data, labels in validation_dataloader:
                labels = labels.to(device=device)
                tokenized_inputs = tokenizer(data, truncation=True, padding='max_length', return_tensors='pt').to(device=device)

                outputs = model(**tokenized_inputs).last_hidden_state.to(device=device)
                loss = contrastive_loss(outputs, labels, tau=tau)
                val_running_loss += loss.item()

            val_avg_loss = val_running_loss / len(validation_dataloader)
            print(
                f"Validation:\tEpoch: {epoch+1},\tLoss: {val_avg_loss}"
            )
        
        if val_avg_loss <= best_loss:
            torch.save(model, f'trained_models/SCL-BERT_{tau}.pth')
            best_loss = val_avg_loss
        else:
            print(f"Best Model is trained for {epoch}")
            break


    



