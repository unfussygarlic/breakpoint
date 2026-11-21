"""
Bayesian Optimization from scratch — numpy only.

Maps onto the three parts of the writeup:
  Part 1  surrogate      -> class GP        (mu, sigma everywhere)
  Part 2  acquisition    -> lcb(...)        (the LCB decision rule)
  Part 3  the loop       -> bottom of file  (fit -> pick -> sim -> refit)

We are MINIMISING J (robustness): low J = a failure, which is what we hunt for.
"""

import numpy as np

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------
# The "expensive simulator". In your real setting this is the robot sim
# that takes a friction value theta and returns a robustness score J.
# Here it's a cheap multimodal function so the script runs instantly.
# Pretend each call costs an hour of compute.
# ----------------------------------------------------------------------
def simulator(theta):
    # one knob: friction. A wiggly curve with a clear global dip (failure).
    return np.sin(theta) + np.sin((10.0 / 3.0) * theta)


DOMAIN = (2.7, 7.5)                       # range of friction values allowed
GRID = np.linspace(*DOMAIN, 400)          # dense grid we evaluate the SURROGATE on
                                          # (cheap — never the simulator)


# ======================================================================
# PART 1 — the surrogate: a Gaussian Process (prediction + error bars)
# ======================================================================
def rbf_kernel(a, b, length=0.8, amp=1.0):
    """Smoothness assumption lives here: nearby thetas -> similar J.
    Returns a matrix of 'how similar' every a is to every b."""
    a = a.reshape(-1, 1)
    b = b.reshape(1, -1)
    sqdist = (a - b) ** 2
    return amp ** 2 * np.exp(-0.5 * sqdist / length ** 2)


class GP:
    def __init__(self, noise=1e-4, length=0.8, amp=1.0):
        self.noise = noise          # tiny jitter: sims are (near) deterministic
        self.length = length
        self.amp = amp

    def fit(self, X, y):
        """X, y = all the black dots we've simulated so far."""
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float)
        K = rbf_kernel(self.X, self.X, self.length, self.amp)
        K += self.noise * np.eye(len(self.X))      # keeps it invertible
        # Cholesky solve == a stable way to compute K^{-1} y
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        return self

    def predict(self, Xs):
        """At every test point return (mu, sigma): best guess + how unsure."""
        Ks = rbf_kernel(self.X, Xs, self.length, self.amp)     # train vs test
        Kss = rbf_kernel(Xs, Xs, self.length, self.amp)        # test vs test

        mu = Ks.T @ self.alpha                                 # posterior mean

        v = np.linalg.solve(self.L, Ks)
        var = np.diag(Kss) - np.sum(v ** 2, axis=0)            # posterior variance
        var = np.clip(var, 1e-12, None)                        # no tiny negatives
        sigma = np.sqrt(var)
        return mu, sigma
        # sigma -> 0 at points we've simulated (band pinches shut),
        # sigma large in the gaps (band bulges). That's the whole point.


# ======================================================================
# PART 2 — the acquisition function: Lower Confidence Bound
#   a(theta) = mu(theta) - kappa * sigma(theta)
#   pick the theta where a is LOWEST.
#   low mu  -> exploit (probably a failure)
#   big sigma -> explore (we're clueless here; subtracting it pulls a down)
# ======================================================================
def lcb(mu, sigma, kappa=2.0):
    return mu - kappa * sigma


# ======================================================================
# PART 3 — the loop, and why it's frugal
# ======================================================================
def bayes_opt(n_init=3, n_steps=8, kappa=2.0):
    # start with a few random black dots
    X = rng.uniform(*DOMAIN, size=n_init).tolist()
    y = [float(simulator(t)) for t in X]

    gp = GP()
    for step in range(n_steps):
        gp.fit(np.array(X), np.array(y))          # 1. fit surrogate to all dots
        mu, sigma = gp.predict(GRID)              #    -> mu, sigma everywhere
        a = lcb(mu, sigma, kappa)                 # 2. score the whole grid (cheap)
        theta_next = GRID[np.argmin(a)]           #    jump to the best next place

        j_next = float(simulator(theta_next))     # 3. ONE expensive sim here
        X.append(float(theta_next))               #    new black dot
        y.append(j_next)                          # 4. (loop) refit next round

        best_i = int(np.argmin(y))
        print(f"step {step+1:2d}: tried theta={theta_next:5.3f}  J={j_next:6.3f}"
              f"   | best so far: theta={X[best_i]:5.3f}  J={y[best_i]:6.3f}")

    best_i = int(np.argmin(y))
    return X[best_i], y[best_i]


if __name__ == "__main__":
    # ground truth, just so we can check BO found the real failure
    true_theta = GRID[np.argmin(simulator(GRID))]
    true_J = simulator(true_theta)

    print("Hunting for the worst-case friction (lowest J)...\n")
    best_theta, best_J = bayes_opt(n_init=3, n_steps=8, kappa=2.0)

    print(f"\nBO found:   theta={best_theta:5.3f}  J={best_J:6.3f}")
    print(f"True worst: theta={true_theta:5.3f}  J={true_J:6.3f}")
    print(f"\nSimulator was called {3 + 8} times total "
          f"(vs 400 for a brute-force grid).")