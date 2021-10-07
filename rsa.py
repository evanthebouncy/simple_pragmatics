from semantics import *
import numpy as np
import matplotlib.pyplot as plt

# take in the set of enumerated inputs and the semantics
# set of utterances, set of hypotheses, the meaning matrix M
def make_meaning(inputs, semantics):
    U = [(x,1) for x in inputs]
    H = []
    M = []
    for semantic in semantics:
        M.append(eval(semantic))
        H.append(semantics[semantic])
    M = np.array(M).transpose()

    return U, H, M

# noralize row / col stuff
def normalise(mat, axis):
    if axis == 0:
        row_sums = mat.sum(axis=1)
        new_matrix = mat / row_sums[:, np.newaxis]
        return new_matrix
    if axis == 1:
        col_sums = mat.sum(axis=0)
        new_matrix = mat / col_sums[np.newaxis, :]
        return new_matrix

# make all the listener and speakers
def make_agents(M):
    S0 = normalise(M, 1)
    L0 = normalise(M, 0)
    S1 = normalise(L0, 1)
    L1 = normalise(S1, 0)
    return S0, L0, S1, L1

# compute P(w'= w) = integrate_u Pspeak(u | w) Plisten(w' | u)
def comm_acc(S,L):
    w_to_w = (S*L).sum(axis=0)
    return w_to_w.mean()

def draw(x, name):
    plt.imshow(x, cmap='gray')
    plt.savefig(f"drawings/{name}.png", dpi=300)
    plt.close()

if __name__ == '__main__':
    # consider all binary string up to length 8
    inputs = enumerate_inputs(8)
    # consider all regex up to length 2, sample 40 of them
    unique_semantics = sample_unique_functions(inputs, 40, func_len=2)

    U, H, M = make_meaning(inputs, unique_semantics)
    print ("all utterances\n", U)
    print ("all hypotheses\n", H)
    print ("meaning matrix\n", M)
    print (M.shape)
    print ("quick check on meaning matrix / lexicon isn't crazy")
    for i in range(10000):
        u_id, h_id = random.choice(range(len(U))), random.choice(range(len(H)))
        u, h = U[u_id], H[h_id]
        function_output = to_func(h)(u[0])
        assert function_output == M[u_id][h_id]
    print ("check passed")


    S0, L0, S1, L1 = make_agents(M)
    draw(M, 'M')
    draw(S0, 'S0')
    draw(L0, 'L0')
    draw(S1, 'S1')
    draw(L1, 'L1')

    s0l0 = comm_acc(S0, L0)
    s1l0 = comm_acc(S1, L0)
    s1l1 = comm_acc(S1, L1)

    print (f"communication accuracies S0-L0 {s0l0} S1-L0 {s1l0} S1-L1 {s1l1}")