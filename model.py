import torch
import torch.nn as nn

class RNNEncoder(nn.Module):
    """
    RNN-based encoder used to encode utterances for lexicon.
    """
    def __init__(self, nfeat, nhid=100, nlayer=1, **rnn_kwargs):
        super(RNNEncoder, self).__init__()
        self.nhid = nhid
        self.nlayer = nlayer
        self.rnn = nn.LSTM(nfeat, nhid, nlayer, batch_first=True, **rnn_kwargs)
    
    def forward(self, inpt, h0=None, device="cpu"):
        # Forward with RNN model. Use specified hidden state, and 0 cell state.
        if h0 is None:
            # Initial hidden state defaults to 0.
            output, (hidden, cell) = self.rnn(inpt)
        else:
            # Use specified hidden state, and 0 cell state.
            B = inpt.size(0) # get batch size
            c0 = torch.zeros((1*self.nlayer, B, self.nhid)).to(device) # default 0s
            output, (hidden, cell) = self.rnn(inpt, (h0, c0))

        return output, hidden, cell

class AG(nn.Module):
    def __init__(self, nU, nH, nhid):
        super(AG, self).__init__()
        # encoder_layer = nn.TransformerEncoderLayer(d_model=nH, nhead=5, batch_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder = RNNEncoder(nH, nhid=nhid)
        self.ff = nn.Sequential(nn.Linear(nhid, nhid), nn.ReLU(), nn.Linear(nhid, nU))

    def forward(self, alts):
        # encode the sequence of inputs (utterance first, then alts -- each of dim nH)
        # (N, T, E) T is the target sequence length, N is the batch size, E is the feature number
        # output = self.encoder(alts)
        _, hidden, _ = self.encoder(alts)
        enc = hidden[-1]
        # enc = output[:, -1, :] # TODO: figure out what to take here
        logits = self.ff(enc) # pass through feedforward readout layer
        return logits
