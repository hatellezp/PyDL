import numpy as np
from PyDL.utils import ncr


def generate_combinations_array(n: int) -> np.ndarray:
    return np.array([ncr(n, r) for r in range(n+1)])


def generate_combinations_matrix(n: int) -> np.ndarray:
    """
    a matrix will all combinations (a b)
    that is m[i, j] = ((i+1) j)
    :param n:
    :return:
    """
    comb = np.zeros([n, n+1])

    for i in range(n):
        comb[i, 0] = 1
        comb[i, i+1] = 1

    for i in range(1, n):
        for j in range(1, i+2):
            comb[i, j] = comb[i-1, j] + comb[i-1, j-1]

    return comb


def seed_generator(arr: np.ndarray) -> Union[None, Generator[np.ndarray, None, None]]:
    n = len(arr)
    k = 0
    for i in range(n):
        if arr[i] == 1:
            k += 1
        elif arr[i] != 0:
            return None

    res = arr
    lb = 0
    limit = n
    while res is not None:
        yield res

        for i in range(n):
            if res[i] == 1:
                lb = i
                break

        if lb == n-k:
            res = None
        else:
            res = next_arr(res, limit, lb, n, k, 0, 0)

