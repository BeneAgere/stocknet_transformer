import torch
from torch import nn
import math


class MLP(nn.Module):

    def __init__(self, embed_dim=8, feature_dim=768, dropout=0.1, device=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_days = 4
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_days * feature_dim, self.embed_dim),
            nn.BatchNorm1d(num_features=self.embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.BatchNorm1d(num_features=self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class AttentiveMLP(nn.Module):

    def __init__(self, embed_dim=8, num_heads=2, feature_dim=768, dropout=0.1, device=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_days = 4
        self.num_heads = num_heads
        self.dropout = dropout
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(num_features=feature_dim),
            nn.Linear(self.num_days * feature_dim, self.embed_dim),
            nn.ReLU()
        )

        self.attn1 = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.dropout, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.attn2 = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.dropout, batch_first=True)

        self.decoder = nn.Sequential(
            nn.BatchNorm1d(num_features=embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        embed = self.encoder(x)
        original_embed = embed.reshape(x.shape[0], 1, -1)
        attended, _ = self.attn1(original_embed, original_embed, original_embed, need_weights=False)

        embed = attended.add(original_embed)
        attended, _ = self.attn2(embed, embed, embed, need_weights=False)
        attended = original_embed.add(attended)

        attended.reshape(x.shape[0], self.embed_dim)
        raw_out = self.decoder(attended)
        return raw_out.reshape(x.shape[0], 1)


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=4, device=None):
        super().__init__()
        self.d_model = d_model
        self.device = device
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < self.d_model:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        pe = pe.to(device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            pe = None
            return x


class TransformerClassifier(nn.Module):

    def __init__(self, embed_dim=8, num_heads=2, feature_dim=768, n_transformer_layers=4, dropout=0.3, input_seq_len=4,
                 output_seq_len=1, aux_target_weight=None, device=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.input_seq_len = input_seq_len
        self.num_transformer_layers = n_transformer_layers
        self.device = device
        self.output_seq_len = output_seq_len
        self.aux_target_weight = aux_target_weight

        self.projection = nn.Linear(self.feature_dim, self.embed_dim)
        self.pe = PositionalEncoder(self.embed_dim, max_seq_len=self.input_seq_len, device=device)
        self.transformer = nn.Transformer(self.embed_dim, self.num_heads, self.num_transformer_layers,
                                          self.num_transformer_layers, batch_first=True, dropout=self.dropout, device=device)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )

    def encoder(self, x):
        projected = self.projection(x)
        pos_encoded_input = self.pe(projected)
        return pos_encoded_input

    def sequential_prediction(self, pos_encoded_input):
        transformer_input = torch.zeros(pos_encoded_input.shape[0], 1, self.embed_dim, device=self.device)
        sequential_predictions = []
        for idx in range(self.output_seq_len):
            transformer_output = self.transformer(pos_encoded_input, transformer_input)
            sequential_predictions.append(self.classifier(transformer_output.squeeze(dim=1)))
            transformer_input = transformer_output
        final_output = torch.cat(sequential_predictions, dim=1)
        pos_encoded_input = transformer_input = transformer_output = class_prediction = sequential_predictions = None
        return final_output

    def forward(self, x):
        pos_encoded_input = self.encoder(x)
        return self.sequential_prediction(pos_encoded_input)


class JointAttentionTransformer(TransformerClassifier):

    def __init__(self, embed_dim=16, num_heads=2, feature_dim=768, n_transformer_layers=4, dropout=0.3, input_seq_len=4,
                         output_seq_len=1, aux_target_weight=0.3, device=None):
        super().__init__(embed_dim, num_heads, feature_dim, n_transformer_layers, dropout, input_seq_len,
                         output_seq_len, aux_target_weight, device)
        self.dim_price = 3
        self.language_model_dim = feature_dim - self.dim_price
        assert embed_dim % 4 == 0
        self.channel_embed_dim = self.embed_dim // 4

        self.projection = None
        self.price_encoder = nn.Sequential(
            nn.Linear(self.dim_price, self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, self.channel_embed_dim),
            nn.ReLU()
        )

        self.tweet_encoder = nn.Sequential(
            nn.Linear(self.language_model_dim, self.language_model_dim // 2),
            nn.ReLU(),
            nn.Linear(self.language_model_dim // 2, self.channel_embed_dim),
            nn.ReLU()
        )

        self.price_attention = nn.MultiheadAttention(self.channel_embed_dim, self.num_heads, dropout=self.dropout, batch_first=True)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.tweet_attention = nn.MultiheadAttention(self.channel_embed_dim, self.num_heads, dropout=self.dropout, batch_first=True)

        self.tweet_attention = nn.MultiheadAttention(self.channel_embed_dim, self.num_heads, dropout=self.dropout,
                                                     batch_first=True)

        self.joint_attention_1 = nn.MultiheadAttention(self.channel_embed_dim, self.num_heads, dropout=self.dropout,
                                                     batch_first=True)
        self.joint_attention_2 = nn.MultiheadAttention(self.channel_embed_dim, self.num_heads, dropout=self.dropout,
                                                     batch_first=True)

        self.pe = PositionalEncoder(self.embed_dim, max_seq_len=4, device=device)
        self.transformer = nn.Transformer(self.embed_dim, self.num_heads, self.num_transformer_layers,
                                          self.num_transformer_layers, batch_first=True, dropout=self.dropout,
                                          device=device)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )

    def encoder(self, x):
        prices = x[:, :, :self.dim_price]
        language_model_activations = x[:, :, self.dim_price:]
        encoded_price = self.price_encoder(prices)
        encoded_tweets = self.tweet_encoder(language_model_activations)
        prices = language_model_activations = None

        attended_price, _ = self.price_attention(encoded_price, encoded_price, encoded_price, need_weights=False)
        attended_tweets, _ = self.tweet_attention(encoded_tweets, encoded_tweets, encoded_tweets, need_weights=False)
        joint_attention_1, _ = self.joint_attention_1(encoded_price, encoded_tweets, encoded_tweets, need_weights=False)
        joint_attention_2, _ = self.joint_attention_2(encoded_tweets, encoded_price, encoded_price, need_weights=False)
        full_attention = torch.cat((attended_price, attended_tweets, joint_attention_1, joint_attention_2), dim=2)
        pos_encoded_input = self.pe(full_attention)
        encoded_price = encoded_tweets = attended_price = attended_tweets = joint_attention_1 = joint_attention_2 = None
        full_attention = None
        return pos_encoded_input
