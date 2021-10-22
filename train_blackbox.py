import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import argparse
import os
import logging
import random
import numpy as np
import rsa
import rsa_alt
from semantics import *

class BlackBoxAG(nn.Module):
    def __init__(self, nU, nhid):
        super(BlackBoxAG, self).__init__()
        self.ff = nn.Sequential(nn.Linear(nU, nhid), nn.ReLU(), nn.Linear(nhid, nU))

    def forward(self, u):
        # u is a onehot vector. output logits over full vocabulary U.
        logits = self.ff(u)
        return logits

def train():
    criterion = nn.CrossEntropyLoss()
    loss_file = os.path.join(args.save, "train-loss.csv")
    rsa_alt.write_line("step,loss,utility,seed", loss_file)
    # for specified # iterations
    for t in range(args.n_iter):
        opt.zero_grad()

        # sample h from P(h) (uniform)
        h = np.random.randint(nH)
        # sample u from S1(*|h)
        S1_dist = S1[:, h]
        u = np.random.choice(range(nU), size=1, p=S1_dist)
        u = torch.tensor(u).to(device)
        # get 1-alt greedy AG as target
        tgt = torch.tensor(greedy_AG[u]).to(device)
        # get distribution from model
        u_onehot = one_hot(u, num_classes=nU).float()
        logits = model(u_onehot)
        loss = criterion(logits, tgt)

        # backward pass
        loss.backward()
        opt.step()

        # save checkpoint
        torch.save((model, opt), args.ckpt)

        # also compute utility, because why not?
        _, utility, _ = evaluate()
        logger.info(f"Step {t}: loss={loss.item()}, utility={utility}")
        rsa_alt.write_line(f"{t},{loss.item()},{utility},{args.seed}", loss_file)
    logger.info("Finished training")

def evaluate():
    # turn AG model into a matrix (sorry for ugly code...)
    with torch.no_grad():
        AG_mat = [
            [model(one_hot(torch.tensor([u]), num_classes=nU).float()).argmax().item()] for u in range(nU)
        ]
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
    _, greedy_AG_utility, greedy_AG = rsa_alt.greedy_search_AG(U, M, S1, AG=None)
    
    # Load model and optimizer checkpoints if specified.
    if os.path.exists(args.ckpt):
        logger.info(f"Initializing model from {args.ckpt}")
        model, opt = torch.load(args.ckpt, map_location=device)
    else:
        logger.info(f"Initializing model from scratch")
        model = BlackBoxAG(nU, args.nhid).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        torch.save((model, opt), args.ckpt)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        train()
    except KeyboardInterrupt:
        logger.warning("="*40)
        logger.warning("Exiting from training early")

    # Use the learned model to generate alts for L1, and evaluate utility.
    model, _ = torch.load(args.ckpt, map_location=device)
    L_alt, utility, AG_mat = evaluate()
    logger.info(f"Final utility of learned AG: {utility}")
    logger.info(f"Utility of greedy AG: {greedy_AG_utility}")
    mask = torch.tensor(greedy_AG).eq(torch.tensor(AG_mat))
    n_agree = sum(mask).item()
    total = mask.numel()
    prop_agree = n_agree / total
    logger.info(f"Agreement between greedy & learned AG: {n_agree}/{total}={prop_agree}")
    logger.info("Greedy AG:")
    logger.info(greedy_AG)
    logger.info("Learned AG:")
    logger.info(AG_mat)