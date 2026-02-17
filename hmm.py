import numpy as np

class HMM:

    def __init__(self, transition_matrix, init_probs):
        self.A = np.log(transition_matrix + 1e-10)
        self.pi = np.log(init_probs + 1e-10)

    def viterbi(self, log_emissions):
        """
        log_emissions shape: (T, 2)
        """

        T = log_emissions.shape[0]
        N = log_emissions.shape[1]

        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        # Initialization
        delta[0] = self.pi + log_emissions[0]

        # Recursion
        for t in range(1, T):
            for j in range(N):
                temp = delta[t-1] + self.A[:, j]
                psi[t, j] = np.argmax(temp)
                delta[t, j] = np.max(temp) + log_emissions[t, j]

        # Backtracking
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])

        for t in reversed(range(T-1)):
            states[t] = psi[t+1, states[t+1]]

        return states
