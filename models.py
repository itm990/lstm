import torch
import torch.nn as nn


# Encoder
class EncoderLSTM(nn.Module):
    def __init__(self, PAD, hidden_size, dict_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(dict_size, hidden_size, padding_idx=PAD)
        self.dropout = nn.Dropout(p=0.1)
        self.lstmcell = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, input_, pre_hidden, pre_cell):
        # batch_size --> (batch_size * hidden_size)
        embedded = self.embedding(input_)
        embedded = self.dropout(embedded)
        hidden, cell = self.lstmcell(embedded, (pre_hidden, pre_cell))
        
        # batch_size --> (batch_size * hidden_size)
        mask = torch.unsqueeze(input_, dim=1).repeat(1, self.hidden_size)
        
        # masking
        hidden = torch.where(mask==0, pre_hidden, hidden)
        cell = torch.where(mask==0, pre_cell, cell)
        
        return hidden, cell

# Decoder
class DecoderLSTM(nn.Module):
    def __init__(self, PAD, hidden_size, dict_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(dict_size, hidden_size, padding_idx=PAD)
        self.dropout = nn.Dropout(p=0.1)
        self.lstmcell = nn.LSTMCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, dict_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_, pre_hidden, pre_cell):
        # batch_size --> (batch_size * hidden_size)
        embedded = self.embedding(input_)
        embedded = self.dropout(embedded)
        hidden, cell = self.lstmcell(embedded, (pre_hidden, pre_cell))
        
        # batch_size --> (batch_size * hidden_size)
        mask = torch.t(input_.repeat(self.hidden_size, 1))

        # masking
        hidden = torch.where(mask==0, pre_hidden, hidden)
        cell = torch.where(mask==0, pre_cell, cell)
        
        output = self.out(hidden)
        output = self.softmax(output)
        
        return output, hidden, cell
