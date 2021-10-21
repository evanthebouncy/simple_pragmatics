import rsa
import random
from semantics import *
import numpy as np
from tqdm import tqdm
# import pandas as pd
import sys

# sample a random 1-hot array
def get_random_1hot(n):
    ret = [0 for i in range(n)]
    ret[random.randint(0,n-1)] = 1
    return ret

def idx_to_onehot(i, n):
    ret = [0 for _ in range(n)]
    ret[i] = 1
    return ret

def get_multi_AG(U, n_alt=1):
    # |U| x n_alt x |U|
    alt_mat = [[get_random_1hot(len(U)) for _ in range(n_alt)] for _ in range(len(U))]
    return alt_mat

def add_AG_rows(alt_mat, U, n_alt):
    # get number of rows to add (n_alt - # rows from original AG matrix)
    n_rows = n_alt - len(alt_mat[0])
    # add random rows to existing AG matrix
    alt_mat = [row + [get_random_1hot(len(U)) for _ in range(n_rows)] for row in alt_mat]
    return alt_mat

# make the alt listener
def get_alt_L(M, AG):
    def alt_L(u):
        # create the small meaning matrix
        first_row = M[u]
        other_rows = M[[a.index(1) for a in AG[u]]] # allows for multiple alts
        rows = [first_row]
        rows.extend(other_rows)
        M_smol = np.array(rows, dtype='float64')
        # get the L1 over the small meaning matrix
        _,L0  = rsa.make_agents(M_smol, prag=False)
        # need to deal with nastiness of all 0 columns
        S1_alt = np.zeros(L0.shape)
        for h_id in range(M_smol.shape[1]):
            if sum(M_smol[:,h_id]) == 0:
                continue
            col = M_smol[:,h_id] / sum(M_smol[:,h_id])
            S1_alt[:,h_id] = col
        L1 = rsa.normalise(S1_alt, 0)
        # the first row is P(h|u)
        return L1[0]
    return alt_L

def comm_acc2(S, alt_L, M):
    h_accs = []
    # for all possible 
    for h_id in range(len(M[0])):
        h_acc = 0
        for u_id, u_prob in enumerate(S[:,h_id]):
            if u_prob < 0.0001:
                continue
            recovered_h_prob = alt_L(u_id)[h_id]
            h_acc += recovered_h_prob * u_prob
        h_accs.append(h_acc)
    return np.mean(h_accs)

def comm_acc_single_u(S, alt_L, M, u_id):
    h_accs = []
    # for all possible 
    for h_id in range(len(M[0])):
        h_acc = 0
        u_prob = S[u_id][h_id] # TODO: double check dims of S
        if u_prob < 0.0001:
            continue
        recovered_h_prob = alt_L(u_id)[h_id]
        h_acc += recovered_h_prob * u_prob
        h_accs.append(h_acc)
    return np.mean(h_accs)

def greedy_search_AG(U,M,S, AG=None):
    print ("greedily constructing AG against speaker")
    # if AG none, then just create 1 alt. otherwise, add single alt to existing AG.
    AG = add_AG_rows_greedy(AG, U, M, S)
    L_alt = get_alt_L(M,AG)
    acc = comm_acc2(S, L_alt, M)
    return L_alt, acc, AG

def add_AG_rows_greedy(alt_mat, U, M, S):
    if alt_mat is None:
        alt_mat = [[] for _ in range(len(U))]
    # find best alt for each utterance
    for u_id, u_alts in enumerate(alt_mat):
        # TODO: should we remove duplicates from the other approaches to make it more fair?
        existing = [a.index(1) for a in u_alts] # indices of existing alts, so we don't add duplicates
        choices = [idx_to_onehot(u, len(U)) for u in range(len(U)) if u not in existing]
        best_acc, best_a = 0, None
        for a in choices:
            # hacky way to construct AG with alternatives only for current u_id
            AG = [[] if i != u_id else alt_mat[u_id]+[a] for i in range(len(U))]
            L_alt = get_alt_L(M,AG)
            # instead of full comm acc, just look at single utterance
            acc = comm_acc_single_u(S, L_alt, M, u_id)
            if acc > best_acc:
                best_acc = acc
                best_a = a
        alt_mat[u_id] += [best_a] # actually extend with best option
    return alt_mat

def random_search_AG(U,M,S, AG=None, n_alt=1):
    best_acc, best_L_alt, best_AG = 0, None, None
    print ("searching for good AG against speaker")
    for i in tqdm(range(100)):
        if AG is None:
            # if no AG specified, then generate from scratch
            AG = get_multi_AG(U, n_alt=n_alt)
        else:
            # add random rows to existing AG
            AG = add_AG_rows(AG, U, n_alt=n_alt)
        L_alt = get_alt_L(M,AG)
        acc = comm_acc2(S, L_alt, M)
        if acc > best_acc:
            best_acc = acc
            best_L_alt = L_alt
            best_AG = AG
    return best_L_alt, best_acc, best_AG

def write_line(l, outfile):
    with open(outfile, "a") as f:
        f.write(l+"\n")

if __name__ == '__main__':
    # example usage: python rsa_alt.py greedy 1111
    AG_type = sys.argv[1]
    SEED = int(sys.argv[2])
    assert AG_type in ["greedy", "sequential", "random"] 
    outfile = f"./drawings/acc_{AG_type}_{SEED}.csv"
    
    # set random seed
    print(f"SEED={SEED}") 
    random.seed(SEED)
    np.random.seed(SEED)

    # consider all binary string up to length 4
    inputs = rsa.enumerate_inputs(6)
    # consider all regex up to length 2, sample 25 of them
    unique_semantics = sample_unique_functions(inputs, 25, func_len=2)

    U, H, M = rsa.make_meaning(inputs, unique_semantics)
    print ("all utterances\n", U)
    print ("all hypotheses\n", H)
    print ("meaning matrix\n", M)
    print (M.shape)

    S0, L0, S1, L1 = rsa.make_agents(M)
    # rsa.draw(M, 'M')
    # rsa.draw(S0, 'S0')
    # rsa.draw(L0, 'L0')
    # rsa.draw(S1, 'S1')
    # rsa.draw(L1, 'L1')

    # accs = dict(acc=[], agent=[], n_alt=[])
    write_line("acc,agent,n_alt,seed", outfile)

    s0l0 = rsa.comm_acc(S0, L0)
    s1l0 = rsa.comm_acc(S1, L0)
    s1l1 = rsa.comm_acc(S1, L1)
    print (f"communication accuracies S0-L0 {s0l0} S1-L0 {s1l0} S1-L1 {s1l1}")
    write_line(f"{s0l0},S0-L0,0,{SEED}", outfile)
    write_line(f"{s1l0},S1-L0,0,{SEED}", outfile)
    write_line(f"{s1l1},S1-L1,{M.shape[0]},{SEED}", outfile)
    # accs["acc"] += [s0l0, s1l0, s1l1]
    # accs["agent"] += ["S0-L0", "S1-L0", "S1-L1"]
    # accs["n_alt"] += [0, 0, M.shape[0]]

    ######## new code below this line ########
    AG = None
    import time
    for n in range(1, M.shape[0]+1, 1):
        start = time.time()
        if AG_type == "greedy":
            # construct AG row-by-row in greedy fashion
            L_alt, acc, AG = greedy_search_AG(U, M, S1, AG=AG)
        else: 
            if AG_type == "random":
                # reset AG each time if we aren't constructing it sequentially
                AG = None
            L_alt, acc, AG = random_search_AG(U,M,S1, AG=AG, n_alt=n)
        end = time.time()
        print (f"communication accuracy S1-L_alt ({n} alts; {(end-start):.4f} sec): {acc}")
        write_line(f"{acc},S1-L_alt,{n},{SEED}", outfile)
        # accs["acc"].append(acc)
        # accs["agent"].append("S1-L_alt")
        # accs["n_alt"].append(n)

    # acc_df = pd.DataFrame(accs)
    # acc_df = acc_df.sort_values(by="n_alt")
    # acc_df["seed"] = SEED
    
    # print(acc_df.head())
    # acc_df.to_csv(outfile, index=False)