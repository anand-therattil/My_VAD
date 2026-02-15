import numpy as np


class DiagonalGMM:

    def __init__(self, n_components=8, max_iter=20, tol=1e-3):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol

    def _initialize(self, X):
        N, D = X.shape

        self.D = D

        # Randomly choose initial means
        indices = np.random.choice(N, self.K, replace=False)
        self.means = X[indices]

        # Initialize variances
        self.vars = np.ones((self.K, D))

        # Uniform weights
        self.weights = np.ones(self.K) / self.K

    def _gaussian_log_prob(self, X):
        N = X.shape[0]
        log_probs = np.zeros((N, self.K))

        for k in range(self.K):
            diff = X - self.means[k]
            var = self.vars[k]

            log_det = np.sum(np.log(var))
            quad = np.sum((diff ** 2) / var, axis=1)

            log_prob = -0.5 * (self.D * np.log(2 * np.pi) + log_det + quad)
            log_probs[:, k] = log_prob

        return log_probs

    def fit(self, X):

        self._initialize(X)
        N, D = X.shape

        prev_log_likelihood = None

        for iteration in range(self.max_iter):

            # E-step
            log_prob = self._gaussian_log_prob(X)
            log_prob += np.log(self.weights + 1e-10)

            # log-sum-exp trick
            max_log = np.max(log_prob, axis=1, keepdims=True)
            log_prob -= max_log
            prob = np.exp(log_prob)
            prob_sum = np.sum(prob, axis=1, keepdims=True)
            responsibilities = prob / prob_sum

            # M-step
            Nk = np.sum(responsibilities, axis=0)

            self.weights = Nk / N
            self.means = (responsibilities.T @ X) / Nk[:, None]

            for k in range(self.K):
                diff = X - self.means[k]
                weighted = responsibilities[:, k][:, None] * (diff ** 2)
                self.vars[k] = np.sum(weighted, axis=0) / Nk[k]

            # Avoid zero variance
            self.vars += 1e-6

            # Compute log likelihood
            log_likelihood = np.sum(np.log(prob_sum) + max_log.flatten())

            print(f"Iteration {iteration+1}, Log Likelihood: {log_likelihood:.2f}")

            if prev_log_likelihood is not None:
                if abs(log_likelihood - prev_log_likelihood) < self.tol:
                    print("Converged!")
                    break
