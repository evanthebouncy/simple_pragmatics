import rsa
import random
from semantics import *
import numpy as np
from tqdm import tqdm
import pandas as pd
from plot import plot_acc_vs_nalt

# sample a random 1-hot array
def get_random_1hot(n):
    ret = [0 for i in range(n)]
    ret[random.randint(0,n-1)] = 1
    return ret

# sample an alt matrix of the form |U|x|U|
# each row is a 
def get_AG(U):
    alt_mat = [get_random_1hot(len(U)) for _ in range(len(U))]
    return alt_mat

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
def get_alt_L(U, M, AG):
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
        L_alt = get_alt_L(U,M,AG)
        acc = comm_acc2(S, L_alt, M)
        if acc > best_acc:
            best_acc = acc
            best_L_alt = L_alt
            best_AG = AG
    return best_L_alt, best_acc, best_AG

if __name__ == '__main__':
    sequential = True # whether or not to generate AG sequentially
    # get ready to track data
    dfs = []

    for SEED in range(10):
        print("="*50)
        print(f"SEED={SEED}")
        print("="*50)
        # set random seed
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

        accs = dict(acc=[], agent=[], n_alt=[])

        s0l0 = rsa.comm_acc(S0, L0)
        s1l0 = rsa.comm_acc(S1, L0)
        s1l1 = rsa.comm_acc(S1, L1)
        print (f"communication accuracies S0-L0 {s0l0} S1-L0 {s1l0} S1-L1 {s1l1}")
        accs["acc"] += [s0l0, s1l0, s1l1]
        accs["agent"] += ["S0-L0", "S1-L0", "S1-L1"]
        accs["n_alt"] += [0, 0, M.shape[0]]

        ######## new code below this line ########
        AG = None
        for n in range(1, M.shape[0]+1, 1):
            if not sequential:
                # reset AG each time if we aren't constructing it sequentially
                AG = None
            L_alt, best_acc, AG = random_search_AG(U,M,S1, AG=AG, n_alt=n)
            print (f"communication accuracy S1-L_alt ({n} alts): {best_acc}")
            accs["acc"].append(best_acc)
            accs["agent"].append("S1-L_alt")
            accs["n_alt"].append(n)

        acc_df = pd.DataFrame(accs)
        acc_df = acc_df.sort_values(by="n_alt")
        acc_df["seed"] = SEED

        # add current df to big list of dfs across seeds
        dfs.append(acc_df)
    
    # concatenate all the dfs (1 per seed)
    df = pd.concat(dfs)
    print(df.head())
    df.to_csv(f"./drawings/acc{'_sequential' if sequential else '_random'}.csv", index=False)

    plot_acc_vs_nalt(df, outpath=f"./drawings/acc{'_sequential' if sequential else '_random'}.png")