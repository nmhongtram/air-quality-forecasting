import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, ahead):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ahead = ahead
        self.output_size = output_size

        # RNN Layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0  # Dropout giữa các layer RNN nếu num_layers > 1
        )

        # Batch Normalization
        self.bn = nn.BatchNorm1d(hidden_size)

        # Dropout Layer
        self.dropout = nn.Dropout(0.3)

        # Improved Output Layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),   
            nn.Linear(hidden_size, output_size * ahead)
        )

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  # out shape: (batch, seq_len, hidden_size)

        # Chỉ lấy hidden state ở bước thời gian cuối cùng
        out = out[:, -1, :]  # shape: (batch, hidden_size)

        # Áp dụng BatchNorm và Dropout
        out = self.bn(out)
        out = self.dropout(out)

        # Dự đoán đầu ra
        out = self.fc(out).view(-1, self.ahead, self.output_size)
        return out
    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, ahead):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ahead = ahead
        self.output_size = output_size

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # Batch Normalization
        self.bn = nn.BatchNorm1d(hidden_size)

        # Dropout Layer
        self.dropout = nn.Dropout(0.3)

        # Output Layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size * ahead)
        )

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch, seq_len, hidden_size)

        # Take the output from the last time step
        out = out[:, -1, :]  # shape: (batch, hidden_size)

        # BatchNorm + Dropout
        out = self.bn(out)
        out = self.dropout(out)

        # Fully connected prediction
        out = self.fc(out).view(-1, self.ahead, self.output_size)
        return out
    


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, ahead):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ahead = ahead
        self.output_size = output_size

        # GRU Layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # Batch Normalization
        self.bn = nn.BatchNorm1d(hidden_size)

        # Dropout Layer
        self.dropout = nn.Dropout(0.3)

        # Output Layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size * ahead)
        )

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # out shape: (batch, seq_len, hidden_size)

        # Take the output from the last time step
        out = out[:, -1, :]  # shape: (batch, hidden_size)

        # BatchNorm + Dropout
        out = self.bn(out)
        out = self.dropout(out)

        # Fully connected prediction
        out = self.fc(out).view(-1, self.ahead, self.output_size)
        return out
