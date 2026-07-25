import torch
from torch import nn
from torch.nn import LSTM
from transformers import BertPreTrainedModel, BertConfig, BertModel
import torch.nn.functional as F

from bert_compat import bert_sequence_and_pooler


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(7)


class DocumentBertSentenceChunkAttentionLSTM(BertPreTrainedModel):
    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertSentenceChunkAttentionLSTM, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.lstm = LSTM(bert_model_config.hidden_size, bert_model_config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, 1),
        )
        self.w_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, bert_model_config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, bert_model_config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, device="cpu", bert_batch_size=0):
        bert_output = torch.zeros(
            size=(
                document_batch.shape[0],
                min(document_batch.shape[1], bert_batch_size),
                self.bert.config.hidden_size,
            ),
            dtype=torch.float,
            device=device,
        )
        for doc_id in range(document_batch.shape[0]):
            _, pooled = bert_sequence_and_pooler(
                self.bert,
                document_batch[doc_id][:bert_batch_size, 0],
                document_batch[doc_id][:bert_batch_size, 1],
                document_batch[doc_id][:bert_batch_size, 2],
            )
            bert_output[doc_id][:bert_batch_size] = self.dropout(pooled)
        output, (_, _) = self.lstm(bert_output.permute(1, 0, 2))
        output = output.permute(1, 0, 2)
        attention_w = torch.tanh(torch.matmul(output, self.w_omega) + self.b_omega)
        attention_u = torch.matmul(attention_w, self.u_omega)
        attention_score = F.softmax(attention_u, dim=1)
        attention_hidden = output * attention_score
        attention_hidden = torch.sum(attention_hidden, dim=1)
        prediction = self.mlp(attention_hidden)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction


class DocumentBertCombineWordDocumentLinear(BertPreTrainedModel):
    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertCombineWordDocumentLinear, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size = 1
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        self.mlp = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size * 2, 1),
        )
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, device="cpu"):
        bert_output = torch.zeros(
            size=(
                document_batch.shape[0],
                min(document_batch.shape[1], self.bert_batch_size),
                self.bert.config.hidden_size * 2,
            ),
            dtype=torch.float,
            device=device,
        )
        for doc_id in range(document_batch.shape[0]):
            sequence_output, pooled_output = bert_sequence_and_pooler(
                self.bert,
                document_batch[doc_id][: self.bert_batch_size, 0],
                document_batch[doc_id][: self.bert_batch_size, 1],
                document_batch[doc_id][: self.bert_batch_size, 2],
            )
            bert_token_max = torch.max(sequence_output, 1)
            bert_output[doc_id][: self.bert_batch_size] = torch.cat(
                (bert_token_max.values, pooled_output), 1
            )

        prediction = self.mlp(bert_output.view(bert_output.shape[0], -1))
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction
