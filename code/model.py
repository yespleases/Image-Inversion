import torch
from torch import nn
from model_config import BertConfig
import math


def expand_attention_mask(mask, dtype=torch.float32):
    mask = mask[:, None, None, :].to(dtype)
    mask = (1.0 - mask) * -10000.0    ###################
    return mask


class Embeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super(Embeddings, self).__init__()

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.type_token_embedding = nn.Embedding(config.type_token_embeddings, config.hidden_size)

        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.emb_drop)

        self.positions = torch.arange(config.max_position_embeddings, dtype=torch.int32, device="cuda")
        self.type_tokens = torch.zeros((1, config.max_position_embeddings), dtype=torch.int32, device="cuda")

    def forward(self, input_ids=None, position_ids=None, type_token_ids=None):

        baz, seq_len = input_ids.shape

        input_emb = self.word_embedding(input_ids)

        if type_token_ids is None:
            type_token_ids = self.type_tokens[:, :seq_len].expand(baz, -1)

        type_token_emb = self.type_token_embedding(type_token_ids)

        input_emb = input_emb + type_token_emb

        if position_ids is None:
            position_ids = self.positions[:seq_len].unsqueeze(0)

        position_emb = self.position_embedding(position_ids)

        embeddings = input_emb + position_emb

        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super(SelfAttention, self).__init__()

        self.att_hidden = config.hidden_size // config.attention_heads
        self.num_heads = config.attention_heads

        self.q_linear = nn.Linear(config.hidden_size, self.att_hidden * self.num_heads)
        self.k_linear = nn.Linear(config.hidden_size, self.att_hidden * self.num_heads)
        self.v_linear = nn.Linear(config.hidden_size, self.att_hidden * self.num_heads)

        self.dropout = nn.Dropout(config.attention_drop)

    def trans_shape(self, x):
        return x.view(x.shape[0], -1, self.num_heads, self.att_hidden)

    def forward(self, hidden_state=None, attention_mask=None):

        q_value = self.q_linear(hidden_state) # b, s, h
        k_value = self.q_linear(hidden_state)
        v_value = self.q_linear(hidden_state)

        q_value = self.trans_shape(q_value) # b s h a
        k_value = self.trans_shape(k_value)
        v_value = self.trans_shape(v_value)

        q_value = q_value.transpose(1, 2)# b h s a
        k_value = k_value.transpose(1, 2).transpose(2, 3)# b h a s
        v_value = v_value.transpose(1, 2)

        attention_scores = torch.matmul(q_value, k_value) # b h s s
        attention_scores = attention_scores / math.sqrt(self.att_hidden)

        if attention_mask is not None:

            attention_scores = attention_scores + attention_mask

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        attention_out = torch.matmul(attention_probs, v_value) # b h s a

        attention_out = attention_out.transpose(1, 2).contiguous()
        attention_out = attention_out.flatten(2)

        return attention_out


class SelfOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super(SelfOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.drop = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_state, input_state):
        x = self.dense(hidden_state)
        x = self.drop(x)
        x = self.norm(x + input_state)
        return x


class Intermediate(nn.Module):
    def __init__(self, config: BertConfig):
        super(Intermediate, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ACT_fc = nn.GELU() if config.hidden_act == "gelu" else nn.ReLU()

    def forward(self, hidden_state):

        x = self.dense(hidden_state)
        x = self.ACT_fc(x)

        return x

class Output(nn.Module):
    def __init__(self, config: BertConfig):
        super(Output, self).__init__()

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.drop = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_state, input_state):
        x = self.dense(hidden_state)
        x = self.drop(x)
        x = self.norm(x + input_state)
        return x


class HeadTransform(nn.Module):
    def __init__(self, config: BertConfig):
        super(HeadTransform, self).__init__()

        self.dense = nn.Linear(config.hidden_size ,config.hidden_size)
        self.ACT_fc = nn.GELU() if config.hidden_act == "gelu" else nn.ReLU()
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_state):
        x = self.dense(hidden_state)
        x = self.ACT_fc(x)
        x = self.norm(x)
        return x


class BertLayer(nn.Module):
    def __init__(self, config:BertConfig):
        super(BertLayer, self).__init__()

        self.self_attention = SelfAttention(config)
        self.self_out = SelfOutput(config)

        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def forward(self, hidden_state=None, attention_mask=None):

        self_atc = self.self_attention(hidden_state, attention_mask=attention_mask)
        self_out = self.self_out(self_atc, hidden_state)

        x = self.intermediate(self_out)

        x = self.output(x, self_out)

        return x

class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertEncoder, self).__init__()

        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.layers)])

    def forward(self, hidden_state=None, attention_mask=None):

        for layer in self.layers:
            hidden_state = layer(hidden_state, attention_mask)

        return hidden_state

class Bert(nn.Module):
    def __init__(self, config: BertConfig):
        super(Bert, self).__init__()

        self.embeddings = nn.Linear(4, config.hidden_size)

        self.encoder = BertEncoder(config)

        self.head_transform = HeadTransform(config)

        #self.out_head = nn.Linear(466*config.hidden_size, 79*79, bias=False)   
        self.out_head = nn.Linear(465*config.hidden_size, 79*79, bias=False)   

        self.out_head.bias = nn.Parameter(torch.zeros(79*79))

        self.norm = nn.LayerNorm(4)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, type_token_ids=None):

        input_ids = self.norm(input_ids)
        embeddings_out = self.embeddings(input_ids)

        if attention_mask is not None:
            attention_mask = expand_attention_mask(attention_mask)

        output = self.encoder(embeddings_out, attention_mask=attention_mask)

        head_out = self.head_transform(output)
        head_out = head_out.flatten(1)

        logits = self.out_head(head_out)
        logits = self.sigmoid(logits)
        logits = logits.view(-1, 79, 79)
        return logits

if __name__ == '__main__':
    config = BertConfig()
    model = Bert(config)

    out = model(torch.ones((2, 269, 4)))    ###################



