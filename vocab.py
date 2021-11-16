import torch

# simple vocab, only 5 possible tokens
PAD_TOK = "[PAD]"
SOS_TOK = "[SOS]"
EOS_TOK = "[EOS]"
ZRO_TOK = "0"
ONE_TOK = "1"
id2tok = [ZRO_TOK, ONE_TOK, SOS_TOK, EOS_TOK, PAD_TOK] # cute ordering so 0s and 1s line up, lol
tok2id = {t: i for i, t in enumerate(id2tok)}
special = [PAD_TOK, SOS_TOK, EOS_TOK]

def tokenize(s):
    tokens = [tok2id[c] for c in s]
    tokens = [tok2id[SOS_TOK]] + tokens + [tok2id[EOS_TOK]]
    return torch.tensor(tokens)

def stringify(token_ids):
    """
    Converts sequence of token indices into a string,
    ignoring special tokens [SOS], [EOS], and [PAD].
    """
    chars = [id2tok[i] for i in token_ids if id2tok[i] not in special]
    return "".join(chars)

def pad(seq_list, pad_idx):
    """
    Pads a list of sequences with specified padding token index.
    """
    padded = torch.nn.utils.rnn.pad_sequence(
        seq_list, batch_first=True, padding_value=pad_idx
    )
    return padded