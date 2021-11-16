import torch
import torch.nn as nn
import argparse
import os
import logging
import random
import numpy as np
import rsa
import rsa_alt
from semantics import *
import pandas as pd

from model import AG

MAX_N_ALT = 1

from vocab import *

def score_alts(u, u_id, alts):
    with torch.no_grad():
        # find indices corresponding to generated alts -- they should be in U
        # first, need to stringify
        alt_strs = [[stringify(a) for a in candidates] for candidates in alts]
        # next, get indices for the ones that aren't empty (all padding)
        alt_ids = [[U.index(a) if a != "" else None for a in candidates] for candidates in alt_strs]
    # compute utility
    batch_utility = []
    for i in range(u.size(0)): 
        AG_mat = [[] for _ in range(nU)]
        utilities = []
        for a in alt_ids[i]:
            if a is not None:
                AG_mat[u_id[i]] = [a] # hacky
                L_alt = rsa_alt.get_alt_L(M, AG_mat) # lazy function
                utility = rsa.utility(S1[np.newaxis, u_id[i]], L_alt(u_id[i])) # single utterance
            else:
                utility = -np.inf
            utilities.append(utility)
        batch_utility.append(utilities)
    return torch.tensor(batch_utility).to(device)

def train():
    criterion = nn.CrossEntropyLoss(ignore_index=tok2id[PAD_TOK], reduction="none")
    loss_file = os.path.join(args.save, "train-loss.csv")
    rsa_alt.write_line("step,n_prev_alt,loss,utility,seed", loss_file)

    # for specified # iterations
    for t in range(args.n_iter):
        opt.zero_grad()

        # sample h from P(h) (uniform). size of h: B
        h = torch.randint(nH, (args.bsz,))
        # sample u from S1(*|h). size of u: B x 1
        S1_dists = torch.tensor([S1[:, h_] for h_ in h]).to(device)
        u_id = torch.multinomial(S1_dists, 1)

        # SIMPLEST SEQ2SEQ MODEL:
        # we want the model to take an utterance and generate a single alternative
        # (TODO: ultimately we want the model to just take an utterance and generate as many alternatives as "needed")
        # tokenize the actual u strings
        u = [tokenize(U[u_]) for u_ in u_id]
        # pad token sequences so lengths match up. size of u: B x T
        u = pad(u, tok2id[PAD_TOK]).to(device)

        # generate a bunch of alternatives (list of len B, each elt being a list of len n_candidates)
        # TODO: should model take the u string itself or L0(u)?
        with torch.no_grad(): # we're kind of evaluating the model here, so no grad
            alts, _, _ = model(u, tok2id[SOS_TOK], tok2id[EOS_TOK], n_sent=args.n_candidate)
            # score all of them
            batch_utility = score_alts(u, u_id, alts)
            # get alts with best scores (1 per batch elt)
            _, best_alt_idxs = torch.max(batch_utility, 1)
            best_alts = [altlist[best_alt_idxs[i]] for i, altlist in enumerate(alts)]

        # use those alts as training signal for the model (*with* grad now)
        best_alts = [a[a != tok2id[PAD_TOK]] for a in best_alts]
        inpt = [a[:-1] for a in best_alts]
        tgt = [a[1:] for a in best_alts]
        # padded_inpt = pad(inpt, tok2id[PAD_TOK]).to(device)
        # padded_tgt = pad(tgt, tok2id[PAD_TOK]).to(device)
        # print("LEN OF INPT:", len(inpt))

        # feed properly to model, with bsz 1. sorry, padding...
        losses = []
        for i in range(args.bsz):
            _, hidden, _ = model.seq2seq.encoder(inpt[i].unsqueeze(0))
            logits, hidden, _ = model.seq2seq.decoder(inpt[i].unsqueeze(0), hidden)
            # Dimensions of `output` are B x L x V, so we need to reshape to B x V x L.
            logits = logits.unsqueeze(0).permute(0, 2, 1)
            loss = criterion(logits, tgt[i].unsqueeze(0))
            losses.append(loss)
        loss = torch.cat(losses, dim=1).mean()
        print(f"{t+1}/{args.n_iter} mean loss:", loss)
        print("="*20)

        # backward pass
        loss.backward()
        opt.step()

        # save checkpoint
        torch.save((model, opt), args.ckpt)

        # log progress at specified interval
        utility = None
        if t % 100 == 0:
            # can evaluate utility if you want, but it slows things down
            # _, utility, _ = evaluate(model)
            logger.info(f"Step {t}: n_prev_alt=1, loss={loss.item()}, utility={utility}")
        rsa_alt.write_line(f"{t},1,{loss.item()},{utility},{args.seed}", loss_file)

    logger.info("Finished training")

def evaluate(model):
    with torch.no_grad():
        AG_mat = []
        for u_id in range(nU):
            # feed u to L0
            u = tokenize(U[u_id]).unsqueeze(0)
            alts, _, _ = model(u, tok2id[SOS_TOK], tok2id[EOS_TOK], n_sent=1)
            alt = alts[0][0]
            alt_str = stringify(alt)
            if alt_str == "":
                logger.info(f"Model alt was all padding for u={u_id}; sampling random alt")
            # NOTE: instead of None, sample randomly... THINK ABOUT THIS.
            alt_id = U.index(alt_str) if alt_str != "" else np.random.randint(nU)
            AG_mat.append([alt_id])
    # compute utility
    L_alt = rsa_alt.get_alt_L(M, AG_mat) # lazy function
    utility = rsa.utility(S1, rsa_alt.matrixify(L_alt, U))
    return L_alt, utility, AG_mat

def init_logger(name=None, debug=False, log_file=None):
    # Basic logging setup.
    logger = logging.getLogger(name)
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    if log_file:
        # Create file handler.
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
    else:
        # Create stream handler that prints to stdout.
        handler = logging.StreamHandler()

    log_format = "%(asctime)s (%(funcName)s) %(levelname)s : %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1111,
                   help="random seed for reproducibility")
    p.add_argument("--n_iter", type=int, default=1000,
                   help="number of training iterations")
    p.add_argument("--bsz", type=int, default=10,
                   help="batch size")
    p.add_argument("--n_candidate", type=int, default=100,
                   help="number of candidate alts to generate")
    p.add_argument("--nhid", type=int, default=50,
                   help="num hidden units")
    p.add_argument("--emb_dim", type=int, default=5,
                   help="embedding dim")
    p.add_argument("--lr", type=float, default=0.001,
                   help="learning rate")
    p.add_argument("--save", type=str, default="./run",
                   help="output folder")
    p.add_argument("--ckpt", type=str, default="model.pt",
                   help="checkpoint to load/save model")
    args = p.parse_args()

    # Set up output directory and logger.
    os.makedirs(args.save)
    log_file = os.path.join(args.save, f"train.log")
    logger = init_logger(log_file=log_file)
    logger.info(vars(args))

    # Manually set seed for reproducibility.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set device to GPU if cuda is available.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Set device to CUDA")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA unvailable; resorting to CPU")

    # consider all binary string up to length 6
    inputs = rsa.enumerate_inputs(6)
    # consider all regex up to length 2, sample 25 of them
    unique_semantics = sample_unique_functions(inputs, 25, func_len=2)

    U, H, M = rsa.make_meaning(inputs, unique_semantics)
    # get rid of tuples in U for simplicity
    U = [u[0] for u in U]
    print ("all utterances\n", U)
    print ("all hypotheses\n", H)
    print ("meaning matrix\n", M)
    nU, nH = M.shape
    print (M.shape)
    np.save(os.path.join(args.save, "U"), U)
    np.save(os.path.join(args.save, "H"), H)
    np.save(os.path.join(args.save, "M"), M)

    # make standard RSA agents and greedy AG
    S0, L0, S1, L1 = rsa.make_agents(M)
    np.save(os.path.join(args.save, "S_0"), S0)
    np.save(os.path.join(args.save, "L_0"), L0)
    np.save(os.path.join(args.save, "S_1"), S1)
    np.save(os.path.join(args.save, "L_1"), L1)

    # read data from previously constructed greedy AG
    greedy_df = pd.read_csv(f"./data/greedy/utility_{args.seed}.csv")
    greedy_utility = {
        k: greedy_df[(greedy_df.n_alt==k)&(greedy_df.agent=="S1-L_alt")].utility.values[0]
        for k in range(1, MAX_N_ALT+1, 1)
    }
    greedy_AG = {
        k: torch.tensor(eval(greedy_df[(greedy_df.n_alt==k)&(greedy_df.agent=="S1-L_alt")].alt_mat.values[0])).to(device)
        for k in range(1, MAX_N_ALT+1, 1)
    }
    greedy_L_alt = rsa_alt.get_alt_L(M, greedy_AG[1])
    np.save(os.path.join(args.save, "L_alt_greedy"), rsa_alt.matrixify(greedy_L_alt, U))

    random_df = pd.read_csv(f"../simple_rsa_coglunch/data/random/utility_{args.seed}.csv")
    random_AG = {
        k: torch.tensor(eval(random_df[(random_df.n_alt==k)&(random_df.agent=="S1-L_alt")].alt_mat.values[0])).to(device)
        for k in range(1, MAX_N_ALT+1, 1)
    }
    random_L_alt = rsa_alt.get_alt_L(M, random_AG[1])
    np.save(os.path.join(args.save, "L_alt_random"), rsa_alt.matrixify(random_L_alt, U))
        
    # Load model and optimizer checkpoints if specified.
    if os.path.exists(args.ckpt):
        logger.info(f"Initializing model from {args.ckpt}")
        model, opt = torch.load(args.ckpt, map_location=device)
    else:
        logger.info(f"Initializing model from scratch")
        model = AG(nU, nH, args.nhid, len(id2tok), args.emb_dim, tok2id[PAD_TOK]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        torch.save((model, opt), args.ckpt)

    print(model)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        train()
    except KeyboardInterrupt:
        logger.warning("="*40)
        logger.warning("Exiting from training early")

    # Use the learned model to generate alts for L1, and evaluate utility.
    model, _ = torch.load(args.ckpt, map_location=device)
    test_file = os.path.join(args.save, "test-utility.csv")
    rsa_alt.write_line("n_prev_alt,utility,AG_type,prop_agree_greedy,seed", test_file)
    n = 0
    greedy = greedy_AG[n+1]
    L_alt, utility, AG_mat = evaluate(model)
    # save L_alt and AG_mat
    np.save(os.path.join(args.save, "L_alt"), rsa_alt.matrixify(L_alt, U))
    np.save(os.path.join(args.save, "alternatives"), AG_mat)
    # hacky: compare only the last column, which is the "added" alt
    AG_mat = torch.tensor(AG_mat)
    assert greedy.size(1) == AG_mat.size(1)
    agree_mask = greedy[:,-1].eq(AG_mat[:,-1])
    prop_agree = agree_mask.sum().item() / agree_mask.numel()
    rsa_alt.write_line(f"{n},{utility},seq2seq,{prop_agree},{args.seed}", test_file)
    rsa_alt.write_line(f"{n},{greedy_utility[n+1]},greedy,1.0,{args.seed}", test_file)