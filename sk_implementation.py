import random
import math
from functools import reduce
from joblib import Parallel, delayed

def a12(lst1, lst2):
    """
    Vargha and Delaney's A12 effect size test.
    Returns the probability that a value from lst1 is greater than a value from lst2.
    """
    more = same = 0.0
    for x in sorted(lst1):
        for y in sorted(lst2):
            if x > y:
                more += 1
            elif x == y:
                same += 1
    n1, n2 = len(lst1), len(lst2)
    return (more + 0.5 * same) / (n1 * n2)

def bootstrap(lst1, lst2, conf=0.01, b=1000, n_jobs=-1, seed=42):
    """
    Bootstrap significance test with reproducible parallel randomness.
    
    """
    def test_stat(y, z):
        y_mu, z_mu = sum(y) / len(y), sum(z) / len(z)
        s1 = math.sqrt(sum((yi - y_mu) ** 2 for yi in y) / (len(y) - 1))
        s2 = math.sqrt(sum((zi - z_mu) ** 2 for zi in z) / (len(z) - 1))
        delta = z_mu - y_mu
        if s1 + s2:
            delta = delta / math.sqrt(s1 / len(y) + s2 / len(z))
        return delta

    y, z = list(lst1), list(lst2)
    x = y + z
    y_mu, z_mu, x_mu = sum(y) / len(y), sum(z) / len(z), sum(x) / len(x)
    tobs = test_stat(y, z)
    yhat = [yi - y_mu + x_mu for yi in y]
    zhat = [zi - z_mu + x_mu for zi in z]

    def boot_one(worker_seed):
        # Create a local RNG with a unique seed for reproducibility in each worker
        rng = random.Random(worker_seed)
        yboot = [rng.choice(yhat) for _ in yhat]
        zboot = [rng.choice(zhat) for _ in zhat]
        return test_stat(yboot, zboot) > tobs

    seeds = [seed + i for i in range(b)]
    bigger = sum(Parallel(n_jobs=n_jobs)(delayed(boot_one)(s) for s in seeds))
    return bigger / b < conf

def scott_knott(groups, cohen=0.3, small=3, use_a12=True, conf=0.01, a12_threshold=0.60, seed=42):
    """
    Scott-Knott clustering for ranking treatments.
    groups: list of (name, [values]) pairs.
    Returns: list of (rank, name, median, iqr, values)
    """
    def median(lst):
        lst = sorted(lst)
        n = len(lst)
        mid = n // 2
        if n % 2:
            return lst[mid]
        return (lst[mid - 1] + lst[mid]) / 2

    def iqr(lst):
        lst = sorted(lst)
        n = len(lst)
        q1 = lst[int(n * 0.25)]
        q3 = lst[int(n * 0.75)]
        return q3 - q1

    def can_split(left, right):
        if len(left) < small or len(right) < small:
            return False
        if use_a12:

            a12_val = a12(right, left)
            if a12_val < a12_threshold:
                return False
            if not bootstrap(left, right, conf=conf, seed=seed):
                return False
        else:
            if abs(median(left) - median(right)) <= cohen * (iqr(left + right) or 1):
                return False
        return True

    def recursive(groups, rank=1, ranks=None):
        if ranks is None:
            ranks = []
        if len(groups) == 1:
            ranks.append((rank, groups[0][0], median(groups[0][1]), iqr(groups[0][1]), groups[0][1]))
            return ranks
        groups = sorted(groups, key=lambda g: median(g[1]))
        all_values = [v for g in groups for v in g[1]]
        best_score, best_split = 0, None
        for i in range(1, len(groups)):
            left = [v for g in groups[:i] for v in g[1]]
            right = [v for g in groups[i:] for v in g[1]]
            if not can_split(left, right):
                continue
            n = len(all_values)
            mu = sum(all_values) / n
            score = (len(left) / n) * (mu - (sum(left) / len(left))) ** 2 + \
                    (len(right) / n) * (mu - (sum(right) / len(right))) ** 2
            if score > best_score:
                best_score, best_split = score, i
        if best_split:
            left_groups = groups[:best_split]
            right_groups = groups[best_split:]
            recursive(left_groups, rank, ranks)
            next_rank = max(r[0] for r in ranks) + 1
            recursive(right_groups, next_rank, ranks)
            return ranks
        else:
            for g in groups:
                ranks.append((rank, g[0], median(g[1]), iqr(g[1]), g[1]))
            return ranks

    return sorted(recursive(groups), key=lambda x: x[0])

# if __name__ == "__main__":
#     data = [
#         ('x1', [0.34, 0.49, 0.51, 0.6]),
#         ('x2', [0.6,  0.7,  0.8,  0.9]),
#         ('x3', [0.15, 0.25, 0.4,  0.35]),
#         ('x4', [0.6,  0.7,  0.8,  0.9]),
#         ('x5', [0.1,  0.2,  0.3,  0.4])
#     ]
#     print("Scott-Knott ranks:")
#     for rank, name, med, iqr_val, values in scott_knott(data, a12_threshold=0.6, seed=42):
#         print(f"Rank {rank}: {name}, median={med}, iqr={iqr_val}, values={values}")
