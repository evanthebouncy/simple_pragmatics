import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, pad_idx, nhid=100, nlayer=1, **rnn_kwargs):
        super(Encoder, self).__init__()
        self.nhid = nhid
        self.nlayer = nlayer
        self.embedding = nn.Embedding(vocab_size, emb_dim, pad_idx)
        self.rnn = nn.LSTM(emb_dim, nhid, nlayer, batch_first=True, **rnn_kwargs)
    
    def forward(self, inpt, h0=None):
        # Embed.
        inpt = self.embedding(inpt)
        # Forward with RNN model. Use specified hidden state, and 0 cell state.
        if h0 is None:
            # Initial hidden state defaults to 0.
            output, (hidden, cell) = self.rnn(inpt)
        else:
            # Use specified hidden state, and 0 (default) cell state.
            c0 = torch.zeros_like(h0)
            output, (hidden, cell) = self.rnn(inpt, (h0, c0))
        return output, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, pad_idx, nlayer=1):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.nhid = hid_dim
        self.nlayer = nlayer
        self.embedding = nn.Embedding(output_dim, emb_dim, pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, nlayer, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
    def forward(self, inpt, hidden, cell=None):
        if cell is None:
            cell = torch.zeros_like(hidden)
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        # inpt = inpt.unsqueeze(0)
        #input = [1, batch size]
        
        embedded = self.embedding(inpt)
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        logits = self.fc_out(output.squeeze(0))
        #logits = [batch size, output dim]
        
        return logits, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        assert encoder.nhid == decoder.nhid, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.nlayer == decoder.nlayer, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, full_inpt, SOS_ID, EOS_ID, n_sent=1, temperature=0.8, alg="sample", max_len=6):
        assert alg in ["greedy", "sample", "gumbel"]
        all_sentences, all_probs, all_logits = [], [], []
        B = full_inpt.size(0) # batch size

        # TODO: make this more efficient, lol
        for i in range(B):
            i_sentences, i_probs, i_logits = [], [], []
            for _ in range(n_sent):
                # Last hidden state of encoder is used as initial hidden state of decoder.
                # hidden = batch_hidden[i] # get one corresponding to batch elt
                # print(f"hidden[{i}]:", hidden.size())
                _, hidden, _ = self.encoder(full_inpt[i].unsqueeze(0))
                # First input is [SOS] for each sentence.
                inpt = torch.tensor([[SOS_ID]]).long().to(hidden.device)
                u_tokens, u_probs, u_full_logits = [inpt.squeeze(0)], [], []
                for _ in range(max_len):
                    # Run decoder conditioned on previous input and hidden state.
                    logits, hidden, _ = self.decoder(inpt, hidden)
                    V_probs = logits.div(temperature).softmax(dim=-1).squeeze()
                    if alg == "greedy":
                        next_token = logits.argmax(dim=-1).squeeze(0)
                    elif alg == "sample":
                        # Non-differentiable sampling.
                        next_token = torch.multinomial(V_probs, num_samples=1)
                    elif alg == "gumbel":
                        # Use Gumbel-Softmax trick.
                        # From documentation: the returned samples will be 
                        # discretized as one-hot vectors, but will be 
                        # differentiated as if it is the soft sample in autograd
                        next_token_onehot = nn.functional.gumbel_softmax(logits, tau=1, hard=True)
                        next_token = next_token_onehot.argmax(dim=-1).squeeze(0)
                    u_probs.append(V_probs[next_token])
                    u_tokens.append(next_token)
                    u_full_logits.append(logits)
                    # If chosen inpt is [EOS], then stop generating.
                    if next_token == EOS_ID:
                        break
                    # Otherwise, update inpt and keep generating.
                    inpt = next_token.unsqueeze(0)
                i_sentences.append(torch.stack(u_tokens).squeeze(-1))
                i_probs.append(torch.stack(u_probs).squeeze(-1))
                i_logits.append(torch.stack(u_full_logits).squeeze(-1))
            all_sentences.append(i_sentences)
            all_probs.append(i_probs)
            all_logits.append(i_logits)

        # Shape: B x n_sent x *
        return all_sentences, all_probs, all_logits

class AG(nn.Module):
    def __init__(self, nU, nH, nhid, vocab_size, emb_dim, pad_idx):
        super(AG, self).__init__()
        encoder = Encoder(vocab_size, emb_dim, pad_idx, nhid=nhid)
        decoder = Decoder(vocab_size, emb_dim, nhid, pad_idx, nlayer=1)
        self.seq2seq = Seq2Seq(encoder, decoder)

    def forward(self, inpt, *args, **kwargs):
        return self.seq2seq(inpt, *args, **kwargs)

# class AG(nn.Module):
#     def __init__(self, nU, nH, nhid):
#         super(AG, self).__init__()
#         self.encoder = RNNEncoder(nH, nhid=nhid)
#         self.readout = nn.Linear(nhid, nU)

#     def forward(self, alts):
#         _, hidden, _ = self.encoder(alts)
#         enc = hidden[-1] # take last hidden layer
#         logits = self.readout(enc)
#         return logits