import rsa
from semantics import *
import numpy as np
from tqdm import tqdm
import random
import sys
# import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# silly helper functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def write_line(l, outfile):
    with open(outfile, "a") as f:
        f.write(l+"\n")

def sample_unique(options, k):
    # sample k elts without replacement
    return np.random.choice(options, k, replace=False).astype(int)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# constructing AG
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# completely random AG,
def get_random_AG(U, n_alt=1):
    # AG is represented as list of numpy arrays: |U| x n_alt
    nU = len(U)
    # make sure that u is not an alternative to itself
    alt_mat = [sample_unique([i for i in range(nU) if i != u], n_alt) for u in range(nU)]
    return alt_mat

# sequential random
def add_AG_rows_random(alt_mat, U, n_alt):
    nU = len(U)
    # get number of rows to add (n_alt - # rows from original AG matrix)
    n_new_rows = n_alt - len(alt_mat[0])
    # find what choices we have so we don't add duplicates
    for u_id, u_alts in enumerate(alt_mat):
        choices = [u for u in range(nU) if u not in u_alts and u != u_id]
        # sample random utterances without replacement, and add to existing AG
        sampled_alts = sample_unique(choices, n_new_rows)
        new_alts = np.concatenate([u_alts, sampled_alts])
        alt_mat[u_id] = new_alts
    return alt_mat

# sequential greedy
def add_AG_rows_greedy(alt_mat, U, M, S):
    nU = len(U)
    if alt_mat is None:
        alt_mat = [[] for _ in range(len(U))]
    # find best alt for each utterance
    for u_id, u_alts in enumerate(alt_mat):
        choices = [u for u in range(nU) if u not in u_alts and u != u_id]
        best_utility, best_a = -np.inf, None
        for a in choices:
            # hacky way to construct AG with alternatives only for current u_id
            AG = [[] if i != u_id else alt_mat[u_id]+[a] for i in range(nU)]
            L_alt = get_alt_L(M,AG)
            # instead of full utility, just look at single utterance
            utility = rsa.utility(S[np.newaxis, u_id], L_alt(u_id))
            if utility > best_utility:
                best_utility = utility
                best_a = a
        alt_mat[u_id] += [best_a] # actually extend with best option
    return alt_mat

# global alts (alts are the same for each u)
def get_global_random_AG(U, n_alt=1):
    # AG is represented as list of numpy arrays: |U| x n_alt
    nU = len(U)
    global_alts = sample_unique(range(nU), n_alt)
    # it's ok if u is an alternative to itself (TODO: address this)
    alt_mat = [global_alts for _ in range(nU)]
    return alt_mat

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# constructing AG-based listener
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# lazy alt listener (function)
def get_alt_L(M, AG):
    def alt_L(u):
        # create the small meaning matrix: rows corresponding to observed u + alts of u
        M_smol = np.concatenate([[M[u]], M[AG[u]]]).astype(np.float64)
        
        # get the L0, S1, and L1 over the small meaning matrix
        _, L0_smol  = rsa.make_agents(M_smol, prag=False)
        # can't use rsa.normalise for S1: need to deal with nastiness of all 0 columns
        S1_alt = np.zeros(L0_smol.shape)
        n_zero_cols = 0
        for h_id in range(M_smol.shape[1]):
            if sum(M_smol[:,h_id]) == 0:
                n_zero_cols += 1
                continue
            col = L0_smol[:,h_id] / sum(L0_smol[:,h_id])
            S1_alt[:,h_id] = col
        L1_alt = rsa.normalise(S1_alt, 0)

        # by construction, the first row is P(h|u)
        return L1_alt[0]
    return alt_L

# matrix-form alt listener
def matrixify(alt_L, U):
    return np.array([alt_L(u_id) for u_id in range(len(U))])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# search algorithms (random/sequential, greedy)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def random_search_AG(U, M, S, sequential=False, AG=None, n_alt=1, n_attempt=100):
    best_utility, best_L_alt, best_AG = -np.inf, None, None
    print ("searching for good AG against speaker")
    for _ in tqdm(range(n_attempt)):
        if sequential and AG is not None:
            # add random rows to existing AG
            cur_AG = add_AG_rows_random(AG, U, n_alt=n_alt)
        else:
            # if completely random, then generate from scratch
            cur_AG = get_random_AG(U, n_alt=n_alt)
        L_alt = get_alt_L(M, cur_AG) # lazy function
        utility = rsa.utility(S, matrixify(L_alt, U))
        if utility > best_utility:
            best_utility = utility
            best_L_alt = L_alt
            best_AG = cur_AG
    return best_L_alt, best_utility, best_AG

def greedy_search_AG(U, M, S, AG=None):
    print ("greedily constructing AG against speaker")
    # if AG none, then just create 1 alt. otherwise, add single alt to existing AG.
    AG = add_AG_rows_greedy(AG, U, M, S)
    L_alt = get_alt_L(M, AG) # lazy function
    utility = rsa.utility(S, matrixify(L_alt, U))
    return L_alt, utility, AG

def global_random_search_AG(U, M, S, n_alt=1, n_attempt=100):
    print ("searching for good *global* AG against speaker")
    best_utility, best_L_alt, best_AG = -np.inf, None, None
    for _ in tqdm(range(n_attempt)):
        # generate from scratch (each row is the same)
        cur_AG = get_global_random_AG(U, n_alt=n_alt)
        L_alt = get_alt_L(M, cur_AG) # lazy function
        utility = rsa.utility(S, matrixify(L_alt, U))
        if utility > best_utility:
            best_utility = utility
            best_L_alt = L_alt
            best_AG = cur_AG
    return best_L_alt, best_utility, best_AG

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# driver code
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    # example usage: python rsa_alt.py greedy 1111
    AG_type = sys.argv[1]
    SEED = int(sys.argv[2])
    assert AG_type in ["greedy", "sequential", "random", "global"] 
    outfile = f"./data/{AG_type}/utility_{SEED}.csv"
    
    # set random seed
    print(f"SEED={SEED}") 
    random.seed(SEED)
    np.random.seed(SEED)
    n_attempt = 100

    # consider all binary string up to length 6
    inputs = rsa.enumerate_inputs(6)
    # consider all regex up to length 2, sample 25 of them
    unique_semantics = sample_unique_functions(inputs, 25, func_len=2)

    U, H, M = rsa.make_meaning(inputs, unique_semantics)
    print ("all utterances\n", U)
    print ("all hypotheses\n", H)
    print ("meaning matrix\n", M)
    print (M.shape)

    # make standard RSA agents and compute utility
    S0, L0, S1, L1 = rsa.make_agents(M)
    # accs = dict(acc=[], agent=[], n_alt=[])
    write_line("utility,utility_ratio,agent,n_alt,alt_mat,seed", outfile)
    s0l0 = rsa.utility(S0, L0)
    s1l0 = rsa.utility(S1, L0)
    s1l1 = rsa.utility(S1, L1)
    print (f"utility: S0-L0 {s0l0}; S1-L0 {s1l0}; S1-L1 {s1l1}")
    write_line(f"{s0l0},{s0l0/s1l1},S0-L0,0,,{SEED}", outfile)
    write_line(f"{s1l0},{s1l0/s1l1},S1-L0,0,,{SEED}", outfile)
    write_line(f"{s1l1},{s1l1/s1l1},S1-L1,{M.shape[0]},,{SEED}", outfile)
    # accs["acc"] += [s0l0, s1l0, s1l1]
    # accs["agent"] += ["S0-L0", "S1-L0", "S1-L1"]
    # accs["n_alt"] += [0, 0, M.shape[0]]

    # perform specified algorithm to build AG
    AG = None
    import time
    for n in range(1, M.shape[0], 1): # max # alts is |U|-1, since u is not an alt to itself
        start = time.time()
        if AG_type == "greedy":
            # construct AG row-by-row in greedy fashion
            L_alt, utility, AG = greedy_search_AG(U, M, S1, AG=AG)
        elif AG_type == "global":
            # construct best global set of alternatives (independent of u)
            L_alt, utility, AG = global_random_search_AG(U, M, S1, n_alt=n, n_attempt=n_attempt)
        else: 
            L_alt, utility, AG = random_search_AG(
                U, M, S1, sequential=(AG_type=="sequential"), AG=AG, n_alt=n, n_attempt=n_attempt
            )
        end = time.time()
        print (f"utility S1-L_alt ({n} alts, {(end-start):.4f} sec): {utility}; S1-L1 ratio {utility/s1l1}")
        alt_mat_str = '"' + str(np.array(AG).tolist()) + '"' if AG is not None else ','
        write_line(f"{utility},{utility/s1l1},S1-L_alt,{n},{alt_mat_str},{SEED}", outfile)
        # accs["acc"].append(acc)
        # accs["agent"].append("S1-L_alt")
        # accs["n_alt"].append(n)

    # check that final utility is equal to full L1 utility
    if utility == s1l1:
        print("achieved full L1 utility :)")
    else:
        print("failed to achieve full L1 utility :(")

    # acc_df = pd.DataFrame(accs)
    # acc_df = acc_df.sort_values(by="n_alt")
    # acc_df["seed"] = SEED
    
    # print(acc_df.head())
    # acc_df.to_csv(outfile, index=False)