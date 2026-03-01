import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

def get_best_hyperparams():
    """Shared utility to load optimized hyperparameters from Config."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Config', 'best_hyperparams.json')
    defaults = {
        "lr": 0.001,
        "hidden_size": 64,
        "dropout": 0.3,
        "batch_size": 64,
        "window_size": 32
    }
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                params = json.load(f)
            return {**defaults, **params}
        except:
            return defaults
    return defaults

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attention_energy = torch.bmm(q, k.transpose(1, 2))
        attention_weights = self.softmax(attention_energy)
        
        context = torch.bmm(attention_weights, v)
        return context, attention_weights

class PatternCNN(nn.Module):
    def __init__(self):
        super(PatternCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten_dim = 64 * 8 * 8
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        x = x.view(-1, self.flatten_dim)
        x = F.relu(self.fc1(x))
        return self.dropout(x)

class TrendLSTM(nn.Module):
    def __init__(self, input_size=27, hidden_size=64, num_layers=2, dropout=0.3):
        super(TrendLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = SelfAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        context, weights = self.attention(out)
        
        final_state = context[:, -1, :] 
        return self.dropout(final_state), weights

class HybridModel(nn.Module):
    def __init__(self, input_size=27, hidden_size=64):
        super(HybridModel, self).__init__()
        self.cnn = PatternCNN()
        self.lstm = TrendLSTM(input_size=input_size, hidden_size=hidden_size)
        
        self.fc_fusion = nn.Linear(128 + hidden_size, 64)
        
        self.fc_cls = nn.Linear(64, 10)
        self.fc_reg = nn.Linear(64, 1)

    def forward(self, gaf_image, seq_data):
        cnn_feat = self.cnn(gaf_image)
        
        lstm_feat, attn_weights = self.lstm(seq_data)
        
        combined = torch.cat((cnn_feat, lstm_feat), dim=1)
        fused = F.relu(self.fc_fusion(combined))
        
        logits = self.fc_cls(fused)
        trend = self.fc_reg(fused)
        
        return logits, trend, attn_weights
