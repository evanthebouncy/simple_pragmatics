import rsa
import random
from semantics import *
import numpy as np
from tqdm import tqdm

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

# make the alt listener
def get_alt_L(U, M, AG):
    def alt_L(u):
        # create the small meaning matrix
        first_row = M[u]
        second_row = M[AG[u].index(1)]
        M_smol = np.array([first_row, second_row], dtype='float64')
        # get the L1 over the small meaning matrix
        _,L0,_,_ = rsa.make_agents(M_smol)
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

def random_search_AG(U,M,S):
    best_acc, best_L_alt = 0, None
    print ("searching for good AG against speaker")
    for i in tqdm(range(100)):
        AG = get_AG(U)
        L_alt = get_alt_L(U,M,AG)
        acc = comm_acc2(S, L_alt, M)
        if acc > best_acc:
            best_acc = acc
            best_L_alt = L_alt
    return best_L_alt, best_acc

if __name__ == '__main__':
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

    s0l0 = rsa.comm_acc(S0, L0)
    s1l0 = rsa.comm_acc(S1, L0)
    s1l1 = rsa.comm_acc(S1, L1)
    print (f"communication accuracies S0-L0 {s0l0} S1-L0 {s1l0} S1-L1 {s1l1}")

    ######## new code below this line ########
    L_alt, best_acc = random_search_AG(U,M,S1)
    print (f"communication accuracy S1-L_alt {best_acc}")
