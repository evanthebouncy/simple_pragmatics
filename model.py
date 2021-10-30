import torch
import torch.nn as nn

class RNNEncoder(nn.Module):
    """
    RNN-based encoder used to encode utterances for lexicon.
    """
    def __init__(self, nfeat, nhid=100, nlayer=2, **rnn_kwargs):
        super(RNNEncoder, self).__init__()
        self.nhid = nhid
        self.nlayer = nlayer
        self.rnn = nn.LSTM(nfeat, nhid, nlayer, batch_first=True, **rnn_kwargs)
    
    def forward(self, inpt, h0=None):
        # Forward with RNN model. Use specified hidden state, and 0 cell state.
        if h0 is None:
            # Initial hidden state defaults to 0.
            output, (hidden, cell) = self.rnn(inpt)
        else:
            # Use specified hidden state, and 0 (default) cell state.
            c0 = torch.zeros_like(h0)
            output, (hidden, cell) = self.rnn(inpt, (h0, c0))
        return output, hidden, cell

class AG(nn.Module):
    def __init__(self, nU, nH, nhid):
        super(AG, self).__init__()
        self.encoder = RNNEncoder(nH, nhid=nhid)
        self.readout = nn.Linear(nhid, nU)

    def forward(self, alts):
        _, hidden, _ = self.encoder(alts)
        enc = hidden[-1] # take last hidden layer
        logits = self.readout(enc)
        return logits
