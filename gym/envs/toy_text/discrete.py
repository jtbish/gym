import numpy as np
from gym import Env, spaces
from gym.utils import seeding


def categorical_sample_naive(prob_n, np_random):
    """
    *** ORIGINAL CATEGORICAL SAMPLE FUNCTION ***
    Bad because it gens the distribution to sample from each time,
    and iterates over whole list of cum. probs to pick an index.

    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


def categorical_sample_fast(csprob_n_with_idxs, np_random):
    """Sample from categorical distribution of transitions with cumulative
    probabilities pre-calced from transitions probs. in desc. order."""
    rand = np_random.rand()
    for (idx, prob) in csprob_n_with_idxs:
        if prob > rand:
            return idx


class DiscreteEnv(Env):
    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)

    (*) dictionary of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS


    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.s = categorical_sample_naive(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample_naive(self.isd, self.np_random)
        self.lastaction = None
        return int(self.s)

    def step(self, a):
        transitions = self.P[self.s][a]
        csprob_n_with_idxs = self.csprob_ns_with_idxs[self.s][a]
        i = categorical_sample_fast(csprob_n_with_idxs, self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})

    def gen_csprob_ns_with_idxs(self):
        """Cache the cumulative next state probabilities *in descending
        order * for each (s, a) pair to make sampling transitions faster.
        Storing in desc. order reduces expected number of iterations through
        cum. probs list each time a sample is drawn."""
        P = self.P
        nS = self.nS
        nA = self.nA
        csprob_ns_with_idxs = {
            s: {a: []
                for a in range(nA)}
            for s in range(nS)
        }
        for s in range(nS):
            for a in range(nA):
                transitions = P[s][a]
                prob_n_with_idxs = list(enumerate([t[0] for t in transitions]))

                # sort probs in desc. order
                sorted_prob_n_with_idxs = \
                    sorted(prob_n_with_idxs, key=lambda tup: tup[1],
                           reverse=True)

                csprob_n_with_idxs = []
                prob_sum = 0.0
                for (idx, prob) in sorted_prob_n_with_idxs:
                    prob_sum += prob
                    csprob_n_with_idxs.append((idx, prob_sum))
                assert prob_sum == 1.0

                csprob_ns_with_idxs[s][a] = csprob_n_with_idxs
        return csprob_ns_with_idxs
