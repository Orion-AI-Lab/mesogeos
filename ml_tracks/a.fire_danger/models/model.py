import torch
import torch.nn as nn
import math


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=24, output_lstm=128, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, output_lstm, num_layers=1, batch_first=True)
        self.ln1 = torch.nn.LayerNorm(input_dim)

        self.fc1 = torch.nn.Linear(output_lstm, output_lstm)
        self.drop1 = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(output_lstm, output_lstm // 2)
        self.drop2 = torch.nn.Dropout(dropout)
        self.fc3 = torch.nn.Linear(output_lstm // 2, 2)

        self.fc_nn = torch.nn.Sequential(
            self.fc1,
            self.drop1,
            self.relu,
            self.fc2,
            self.drop2,
            self.relu,
            self.fc3
        )

    def forward(self, x):
        x = self.ln1(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc_nn(lstm_out[:, -1, :])
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 256, dropout: float = 0.1, max_len: int = 30):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerNet(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(
            self,
            seq_len=30,
            input_dim=24,
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            num_layers=4,
            dropout=0.1,
            activation="relu",
            classifier_dropout=0.1,
            channel_attention=False
    ):

        super().__init__()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        # self.emb = nn.Embedding(input_dim, d_model)
        self.channel_attention = channel_attention

        self.lin_time = nn.Linear(input_dim, d_model)
        self.lin_channel = nn.Linear(seq_len, d_model)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout
        )

        encoder_layer_time = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder_time = nn.TransformerEncoder(
            encoder_layer_time,
            num_layers=num_layers,
        )

        encoder_layer_channel = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder_channel = nn.TransformerEncoder(
            encoder_layer_channel,
            num_layers=num_layers,
        )

        self.out_time = nn.Linear(d_model, d_model)
        self.out_channel = nn.Linear(d_model, d_model)

        self.lin = nn.Linear(d_model * 2, 2)

        if self.channel_attention:
            self.classifier = nn.Linear(d_model * 2, 2)
        else:
            self.classifier = nn.Linear(d_model, 2)

        self.d_model = d_model

    def resh(self, x, y):
        return x.unsqueeze(1).expand(y.size(0), -1)

    def forward(self, x_):

        x = torch.tanh(self.lin_time(x_))
        x = self.pos_encoder(x)
        x = self.transformer_encoder_time(x)
        x = x[0, :, :]

        if self.channel_attention:
            y = torch.transpose(x_, 0, 2)
            y = torch.tanh(self.lin_channel(y))
            y = self.transformer_encoder_channel(y)

            x = torch.tanh(self.out_time(x))
            y = torch.tanh(self.out_channel(y[0, :, :]))

            h = self.lin(torch.cat([x, y], dim=1))

            m = nn.Softmax(dim=1)
            g = m(h)

            g1 = g[:, 0]
            g2 = g[:, 1]

            x = torch.cat([self.resh(g1, x) * x, self.resh(g2, x) * y], dim=1)

        x = self.classifier(x)

        return x
