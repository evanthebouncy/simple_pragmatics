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
from copy import deepcopy

from model import AG

MAX_N_ALT = 20

def train():
    criterion = nn.CrossEntropyLoss()
    loss_file = os.path.join(args.save, "train-loss.csv")
    rsa_alt.write_line("step,n_prev_alt,loss,utility,seed", loss_file)
    # for specified # iterations
    for t in range(args.n_iter):
        opt.zero_grad()

        # sample h from P(h) (uniform)
        h = np.random.randint(nH)
        # sample u from S1(*|h)
        S1_dist = S1[:, h]
        u = np.random.choice(range(nU), p=S1_dist)

        # randomly sample n (number of previous alternatives)
        n = np.random.randint(1, MAX_N_ALT)
        # get previous alternatives from greedy AG for n
        prev_alts = torch.tensor(greedy_AG[n][u]).to(device)
        # get greedy AG prediction at n+1 as target
        tgt = torch.tensor(greedy_AG[n+1][u][-1]).to(device).unsqueeze(0)
        # create inputs by feeding u and prev_alts to L0
        L0_u = torch.from_numpy(L0[u]).unsqueeze(0).float()
        L0_prev = torch.from_numpy(L0[prev_alts]).float()
        if n == 1:
            L0_prev = L0_prev.unsqueeze(0)
        # put u last so model doesn't have to keep track of long dependency
        inpt = torch.cat([L0_prev, L0_u], dim=0).unsqueeze(0).to(device) 

        # forward pass
        logits = model(inpt)
        loss = criterion(logits, tgt)

        # backward pass
        loss.backward()
        opt.step()

        # save checkpoint
        torch.save((model, opt), args.ckpt)

        # log progress at specified interval
        utility = None
        if t % 100 == 0:
            # can evaluate utility if you want, but it slows things down
            # _, utility, _ = evaluate(model, n)
            logger.info(f"Step {t}: n_prev_alt={n}, loss={loss.item()}, utility={utility}")
        rsa_alt.write_line(f"{t},{n},{loss.item()},{utility},{args.seed}", loss_file)
    logger.info("Finished training")

def evaluate(model, n):
    with torch.no_grad():
        AG_mat = []
        for u in range(nU):
            # get previous alternatives from greedy AG for n
            prev_alts = deepcopy(greedy_AG[n][u])
            AG_mat.append(prev_alts)
            # create inputs by feeding u and prev_alts to L0
            L0_u = torch.from_numpy(L0[u]).unsqueeze(0).float()
            L0_prev = torch.from_numpy(L0[prev_alts]).float()
            inpt = torch.cat([L0_prev, L0_u]).unsqueeze(0).to(device)
            logits = model(inpt)
            pred_alt = logits.argmax().item()
            AG_mat[u].append(pred_alt)
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
                   help="number of training examples")
    p.add_argument("--nhid", type=int, default=50,
                   help="num hidden units")
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
    print ("all utterances\n", U)
    print ("all hypotheses\n", H)
    print ("meaning matrix\n", M)
    nU, nH = M.shape
    print (M.shape)

    # make standard RSA agents and greedy AG
    S0, L0, S1, L1 = rsa.make_agents(M)

    # read data from previously constructed greedy AG
    greedy_df = pd.read_csv(f"./data/greedy/utility_{args.seed}.csv")
    greedy_utility = {
        k: greedy_df[(greedy_df.n_alt==k)&(greedy_df.agent=="S1-L_alt")].utility.values[0]
        for k in range(1, MAX_N_ALT+1, 1)
    }
    greedy_AG = {
        k: eval(greedy_df[(greedy_df.n_alt==k)&(greedy_df.agent=="S1-L_alt")].alt_mat.values[0])
        for k in range(1, MAX_N_ALT+1, 1)
    }
        
    # Load model and optimizer checkpoints if specified.
    if os.path.exists(args.ckpt):
        logger.info(f"Initializing model from {args.ckpt}")
        model, opt = torch.load(args.ckpt, map_location=device)
    else:
        logger.info(f"Initializing model from scratch")
        model = AG(nU, nH, args.nhid).to(device)
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
    for n in range(1, MAX_N_ALT, 1):
        greedy = torch.tensor(greedy_AG[n+1])
        L_alt, utility, AG_mat = evaluate(model, n)
        # hacky: compare only the last column, which is the "added" alt
        AG_mat = torch.tensor(AG_mat)
        assert greedy.size(1) == AG_mat.size(1)
        agree_mask = greedy[:,-1].eq(AG_mat[:,-1])
        prop_agree = agree_mask.sum().item() / agree_mask.numel()
        rsa_alt.write_line(f"{n},{utility},learned,{prop_agree},{args.seed}", test_file)
        rsa_alt.write_line(f"{n},{greedy_utility[n+1]},greedy,1.0,{args.seed}", test_file)