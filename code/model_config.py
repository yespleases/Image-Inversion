import json


class BertConfig:
    def __init__(self):
        self.vocab_size = 1000
        self.hidden_size = 256
        self.intermediate_size = 1024
        self.layers = 6
        # self.max_position_embeddings = 512
        # self.type_token_embeddings = 2
        self.emb_drop = 0.1
        self.attention_heads = 8
        self.attention_drop = 0.0
        self.dropout_rate = 0.1
        self.hidden_act = "gelu"

    @classmethod
    def from_json_file(cls, file):

        config = BertConfig()
        with open(file, "r") as f:
            config_dict = json.load(f)

        for key in list(config_dict.keys()):
            config.__dict__[key] = config_dict[key]

        return config

    def __str__(self):
        return str(self.__dict__)
