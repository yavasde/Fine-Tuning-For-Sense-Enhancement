import torch
import torch.nn as nn
from transformers import BertModel


class BERTforSCL(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', drop_out=0.1):
        super(BERTforSCL, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, hidden_dropout_prob=drop_out)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs

class BERTforSPL(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2, drop_out=0.3):
        super(BERTforSPL, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, hidden_dropout_prob=drop_out)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)  

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = outputs.last_hidden_state   
        logits = self.classifier(x)  
        return logits

class WordOccurrenceClassifier(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", drop_out=0.7):
        super(WordOccurrenceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, hidden_dropout_prob=drop_out)
        self.classifier = nn.Linear(2 * 768, 1)

    def forward(self, tokenized_sentences, word_indices):
        input_ids = tokenized_sentences['input_ids']
        token_type_ids = tokenized_sentences['token_type_ids']
        attention_mask = tokenized_sentences['attention_mask']

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state

        emd1s = []
        for i in range(len(word_indices)):
            if len(word_indices[i][0]) == 1:
                emd1s.append(hidden_states[i][word_indices[i][0][0]])
            elif len(word_indices[i][0]) > 1:
                subword_embeddings = []
                for ind in word_indices[i][0]:
                    subword_embeddings.append(hidden_states[i][ind])
                emd1s.append(torch.mean(torch.stack(subword_embeddings), dim=0))
                       
        emd2s = []
        for i in range(len(word_indices)):
            if len(word_indices[i][1]) == 1:
                emd2s.append(hidden_states[i][word_indices[i][1][0]])  
            elif len(word_indices[i][1]) > 1:
                subword_embeddings = []
                for ind in word_indices[i][1]:
                    subword_embeddings.append(hidden_states[i][ind])
                emd2s.append(torch.mean(torch.stack(subword_embeddings), dim=0))

        combined_emb = torch.cat([torch.stack(emd1s), torch.stack(emd2s)], dim=-1)
        logits = self.classifier(combined_emb)
        return logits