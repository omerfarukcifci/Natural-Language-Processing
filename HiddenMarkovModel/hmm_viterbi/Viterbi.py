import pandas as pd
import numpy as np


class Trellis():
    def __init__(self):
        self.viterbi = int()
        self.backpointer = int()

def viterbi(tags,words,init,trans,emit):

    trellis = [[Trellis() for x in range(len(tags))] for y in range(len(words))]
    p_init = init
    p_trans = trans
    p_emit = emit
    for t in range(1,len(tags)):
        # print(p_init.get(tags[t]))
        trellis[1][t].viterbi = p_init.get(tags[t]) * p_emit.get(words[0]+'|'+tags[t],0)
        # print(tags[t]," ",trellis[1][t].viterbi)

    for i in range(2,len(words)):
        for t in range(1,len(tags)):
            trellis[i][t] = Trellis()
            for t2 in range(1,len(tags)):
                tmp = trellis[i-1][t2].viterbi * p_trans.get(tags[t2]+'|'+tags[t])
                if(tmp > trellis[i][t].viterbi):
                    trellis[i][t].viterbi = tmp
                    trellis[i][t].backpointer = t2
            trellis[i][t].viterbi *= p_emit.get(words[i-1]+'|'+tags[t],0)

    t_max = None
    vit_max = 0

    for t in range(1,len(tags)):
        if(trellis[len(words)-1][t].viterbi > vit_max):
            t_max = t
            vit_max = trellis[len(words)-1][t].viterbi
    if(t_max != None):
        print(backtrace(len(words),t_max,trellis,tags))
        return backtrace(len(words),t_max,trellis,tags)
    # return unpack(len(words),t_max,trellis,tags)


def backtrace(n,t,trellis,tag):
    i = n-1
    tags = []
    while(i>=0):
        tags.append(tag[t])
        t = trellis[i][t].backpointer
        i-=1
    return tags[::-1]

# def forward(V, a, b, initial_distribution):
#     alpha = np.zeros((V.shape[0], a.shape[0]))
#     alpha[0, :] = initial_distribution * b[:, V[0]]
#
#     for t in range(1, V.shape[0]):
#         for j in range(a.shape[0]):
#             # Matrix Computation Steps
#             #                  ((1x2) . (1x2))      *     (1)
#             #                        (1)            *     (1)
#             alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]
#
#     return alpha
#
#
# def backward(V, a, b):
#     beta = np.zeros((V.shape[0], a.shape[0]))
#
#     # setting beta(T) = 1
#     beta[V.shape[0] - 1] = np.ones((a.shape[0]))
#
#     # Loop in backward way from T-1 to
#     # Due to python indexing the actual loop will be T-2 to 0
#     for t in range(V.shape[0] - 2, -1, -1):
#         for j in range(a.shape[0]):
#             beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])
#
#     return beta
#
#
# def baum_welch(V, a, b, initial_distribution, n_iter=100):
#     M = a.shape[0]
#     T = len(V)
#
#     for n in range(n_iter):
#         alpha = forward(V, a, b, initial_distribution)
#         beta = backward(V, a, b)
#
#         xi = np.zeros((M, M, T - 1))
#         for t in range(T - 1):
#             denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
#             for i in range(M):
#                 numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
#                 xi[i, :, t] = numerator / denominator
#
#         gamma = np.sum(xi, axis=1)
#         a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
#
#         # Add additional T'th element in gamma
#         gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
#
#         K = b.shape[1]
#         denominator = np.sum(gamma, axis=1)
#         for l in range(K):
#             b[:, l] = np.sum(gamma[:, V == l], axis=1)
#
#         b = np.divide(b, denominator.reshape((-1, 1)))
#
#     return (a, b)
#
#
# def viterbi(V, a, b, initial_distribution):
#     T = V.shape[0]
#     M = a.shape[0]
#
#     omega = np.zeros((T, M))
#     omega[0, :] = np.log(initial_distribution * b[:, V[0]])
#
#     prev = np.zeros((T - 1, M))
#
#     for t in range(1, T):
#         for j in range(M):
#             # Same as Forward Probability
#             probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])
#
#             # This is our most probable state given previous state at time t (1)
#             prev[t - 1, j] = np.argmax(probability)
#
#             # This is the probability of the most probable state (2)
#             omega[t, j] = np.max(probability)
#
#     # Path Array
#     S = np.zeros(T)
#
#     # Find the most probable last hidden state
#     last_state = np.argmax(omega[T - 1, :])
#
#     S[0] = last_state
#
#     backtrack_index = 1
#     for i in range(T - 2, -1, -1):
#         S[backtrack_index] = prev[i, int(last_state)]
#         last_state = prev[i, int(last_state)]
#         backtrack_index += 1
#
#     # Flip the path array since we were backtracking
#     S = np.flip(S, axis=0)
#
#     # Convert numeric values to actual hidden states
#     result = []
#     for s in S:
#         if s == 0:
#             result.append("A")
#         else:
#             result.append("B")
#
#     return result