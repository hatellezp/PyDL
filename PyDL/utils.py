import numpy as np

from bitarray import bitarray

from typing import Generator, List, Union, Optional, Tuple
import operator as op
from functools import reduce

import time


# TODO: verify that this works!!!
def ncr(n, k):
    """
    combinations of k elements in
    :param n:
    :param k:
    :return:
    """
    r = min(k, n-k)
    numerator = reduce(op.mul, range(n, n-k, -1), 1)
    denominator = reduce(op.mul, range(1, k+1), 1)
    return numerator // denominator


def max_ncr(n):
    return ncr(n, n // 2)


def not_empty_subsets(n, k):
    if k > n:
        return 2**n - 1
    else:
        return reduce(op.add, [ncr(n, i) for i in range(1, k+1)], 0)


def create_generator(n: int, k: int, method) -> \
        Union[Generator[bitarray, None, None],
              Generator[np.ndarray, None, None]]:
    """
    this function produces a generator
    that will return in lexicographic order the subsets of k elements of a
    set of n elements
    :param n:
    :param k:
    :param method:
    :return:
    """

    if k == 0:
        done = False
        if done:
            yield None
        else:
            done = True

            if method == 'numpy':
                res = np.zeros(n, dtype=np.int)
            elif method == 'bitarray':
                res = bitarray([False for i in range(n)])

            yield res
    else:
        if method == 'numpy':
            res = np.array([(lambda x: 1 if x < k else 0)(i) for i in range(n)])
        elif method == 'bitarray':
            res = bitarray([(lambda x: x < k)(i) for i in range(n)])

        limit = n
        lb = 0

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


# helper function for the create_generator
def next_arr(arr: Union[np.ndarray, bitarray], limit: int, mlb: int, mn: int, mk: int,
             lc: int, oe: int) -> Union[np.ndarray, bitarray]:
    # first base case
    if mlb == limit-1 and arr[mlb] == 1:
        counter = oe + 2
        arr[lc] = 0

        for i in range(lc+1, limit):
            if counter > 0:
                arr[i] = 1
                counter -= 1
            else:
                arr[i] = 0

        return arr

    if mlb == limit-1 and arr[mlb] == 0:

        counter = oe + 1
        arr[lc] = 0

        for i in range(lc+1, limit):
            if counter > 0:
                arr[i] = 1
                counter -= 1
            else:
                arr[i] = 0

        return arr

    else:
        lc_changed = False

        if arr[mlb] == 1 and arr[mlb+1] == 0:
            lc = mlb
            oe = 0
            lc_changed = True

        if arr[mlb] == 0:
            return next_arr(arr, limit, mlb+1, mn-1, mk, lc, oe)
        else:
            if lc_changed:
                return next_arr(arr, limit, mlb + 1, mn - 1, mk, lc, oe)
            else:
                return next_arr(arr, limit, mlb+1, mn-1, mk-1, lc, oe+1)


def banker_sequence(n, k, method='bitarray'):
    """
    Algorithm from 'Efficiently Enumerating the Subsets of a Set'.
    The interesting part is that it generates the subsets of a set
    in an increasingly manner.

    :param n: size of the set
    :param k: size of the subset
    :param method: use numpy or bitstring as lib to generate arrays
    :return: a generator for the subsets of size k of a set of size n ordered by
             lexicographic order
    """

    return create_generator(n, k, method)

# I forgot why 'gac'
def gac(x: Union[bitarray, int], n: Optional[int] = None) -> Union[bitarray, int]:
    """
    this function will take as argument a bitarray or an int
    and ouput the equivalent in the other type
    the short name is intended
    :param n:
    :param x:
    :return:
    """

    if isinstance(x, int) and n is None:
        raise TypeError("'n' argument missing when attempting to transform int into bitarray")

    if isinstance(x, int):
        unpadded = [int(digit) for digit in bin(x)[2:]]
        l = len(unpadded)

        if l > n:
            raise Exception("n must be larger or equal than the bitarray created")

        res = bitarray([
            (lambda v: 0 if v < (n - l) else unpadded[v - (n - l)])(i)
            for i in range(n)
        ])

        return res
    elif isinstance(x, bitarray):
        l = len(x)
        res = 0
        for i in range(l, 0, -1):
            res += int(x[i - 1]) * 2 ** (l - i)

        return res
    else:
        raise TypeError("'x' argument of type '{}' expected bitarray or int".format(type(x)))


def find_all_not_empty_subsets(b: Union[bitarray, int], n: Optional[int] = None) \
        -> Union[List[bitarray], List[int]]:

    if isinstance(b, int) and n is None:
        raise TypeError("'n' argument missing when attempting to analyse int")

    if isinstance(b, int):
        pass
    elif isinstance(b, bitarray):
        pass
    else:
        raise TypeError("'b' argument of type {}, expected int or bitarray".format(type(b)))


def bit_diff(x: bitarray, y: bitarray, detailed: bool = False) -> Tuple[List[int], int]:
    """
    computes the difference between two bitarrays modeling subsets of a set

    :param x: first bitarray
    :param y: second bitarray
    :param detailed: if a list of the detailed difference should be returned
    :return:  a tuple: l,d where l is a list of the detailed difference and
              d values -1 if y is bigger, 1 if x is bigger, 0 if they are the
              same or 2 if they are incomparable
    """
    xl = len(x)
    yl = len(y)

    if xl != yl:
        raise ValueError("mismatched length, 'x' lenght: {}, 'y' length: {}".format(xl, yl))
    else:
        sign = 0
        res = []
        for i in range(xl):
            d = x[i] - y[i]

            # first append the value
            # if you want only to know the relation then forget about the list
            if detailed:
                res.append(d)

            # now the superset or subset logic
            if sign == 2 or d == 0:
                continue
            else:
                if d == 1 and sign == -1:
                    sign = 2
                elif d == -1 and sign == 1:
                    sign = 2
                elif sign == 0:
                    sign = d
                else:
                    continue

        return res, sign


if __name__ == "__main__":
    print("Hello World from the utils module!")


    def useless_function(a, t=0):
        time.sleep(t)
        return None

    def time_tracker(n, k, t, method='bitarray'):
        tim = time.perf_counter()

        for i in range(k):
            gen = banker_sequence(n, i, method=method)
            for a in gen:
                useless_function(a, t*0.001)

        tim = time.perf_counter() - tim

        return tim

    a = bitarray(bin(10)[2:])
    b = bitarray(bin(11)[2:])
    c = bitarray(bin(14)[2:])

    print("a", a)
    print('b', b)
    print("c", c)

    print("a-b", bit_diff(a, b))
    print("b-a", bit_diff(b, a))
    print("a-c", bit_diff(a, c))
    print("c-a", bit_diff(c, a))
    print("b-c", bit_diff(b, c))
    print("c-b", bit_diff(c, b))

    print("a-a", bit_diff(a, a))



