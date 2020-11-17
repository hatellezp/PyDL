from collections import OrderedDict
from bitarray import bitarray

from typing import Union, Optional, List, Generator

from utils import gac, bit_diff


class SOD:  # stands for Subset OrderedDict
    """
    how this works: this is nothing else than a wrapper around an ordered dict
    with some other stuff

    the keys are bitarrays modeling subsets of a bigger set
    the values stored are True or False for consistency or inconsistency
    respectively.

    two things can happen here:
        - suppose you add a new entry with value False, meaning inconsistency
          then, whenever you try to add a new entry modeling a superset of this
          set will be not necessary, as you already know the answer
        - suppose you add a new entry with value True, then all of the already
          existent entries which model subsets of this set should be deleted,
          as its consistency it is now implied by the new entry
    """
    def __init__(self, size: int):
        self.size = size
        self.od = OrderedDict()

    def __repr__(self):
        res = "["
        for key, item in self.od.items():
            bit_key = gac(key, self.size)
            res += "({}, {}) -> {}, ".format(key, bit_key, item)

        res += "]"
        return res

    def _find_sets(self, x: bitarray, sub_or_super: int, proper=True) -> List[bitarray]:
        """
        :param x: the subset in question
        :param sub_or_super:  if we are to find supersets (1) or subsets (-1)
        :param proper: flag to consider only proper supersets and subsets
        :return:  a list of sets as bitarrays
        """

        outcomes = [1] if proper else [0, 1]

        if sub_or_super == -1:  # to find subsets
            return [key for key in self.od.keys() if bit_diff(x, gac(key, self.size))[1] in outcomes]
        elif sub_or_super == 1:    # to find supersets
            return [key for key in self.od.keys() if bit_diff(gac(key, self.size), x)[1] in outcomes]
        else:
            raise ValueError(("'sub_or_super' argument must be 1 for "
                              "supersets or -1 for subsets, found {}")
                             .format(sub_or_super))

    def _find_sets_generator(self, x: bitarray, sub_or_super: int) -> Generator[bitarray, None, None]:
        if sub_or_super == -1:  # to find subsets
            for key in self.od:
                if bit_diff(x, gac(key, self.size))[1] in [0, 1]:
                    yield key
        elif sub_or_super == 1:    # to find supersets
            for key in self.od:
                if bit_diff(gac(key, self.size), x)[1] in [0, 1]:
                    yield key
        else:
            raise ValueError(("'sub_or_super' argument must be 1 for "
                              "supersets or -1 for subsets, found {}")
                             .format(sub_or_super))

    def __setitem__(self, key: Union[int, bitarray], value: bool) -> None:
        int_key = key if isinstance(key, int) else gac(key, self.size)
        bit_key = key if isinstance(key, bitarray) else gac(key, self.size)

        if value:  # because of consistency, all subsets can be deleted
            # TODO: verify this for the love of god !!!
            for subset in self._find_sets(bit_key, -1):
                del self.od[subset]

            # now you can safely add the new value
            self.od[int_key] = value

        elif not value:  # because of inconsistency, remember tha that next supersets are not needed

            key_can_be_deduced = False
            for subset in self._find_sets(bit_key, -1):
                try:
                    if not self[subset]:
                        key_can_be_deduced = True
                        break
                except KeyError:
                    pass

            if not key_can_be_deduced:
                self.od[int_key] = value

    def __getitem__(self, key: Union[int, bitarray]) -> bool:
        int_key = key if isinstance(key, int) else gac(key, self.size)

        try:
            return self.od[int_key]
        except KeyError:
            print("SOD: {} not found as key, attempting to deduce value.".format(key))
            bit_key = key if isinstance(key, bitarray) else gac(key, self.size)

            # TODO: no need to have them all at once, better is a generator
            #     : and if we find an answer then we return

            # be lazy, compute only if needed
            # UPDATE: generators can't be used because they force a runtime mutation and python
            #         doens't like it

            for subset in self._find_sets(bit_key, -1):
                try:
                    consistent = self[subset]
                    if not consistent:
                        return consistent
                except KeyError:
                    pass

            # no answer found in subsets, search in the supersets
            for superset in self._find_sets(bit_key, 1):
                try:
                    consistent = self[superset]
                    if consistent:
                        return consistent
                except KeyError:
                    pass

            # nothing was found, raise Exception
            raise KeyError("SOD: your key {} doesn't exist and can't be deduced".format(key))


if __name__ == "__main__":
    print("hello there from the SOD module!")

    def print_sod(sod):
        for item in sod.od.items():
            print(item)

    n = 4


    bits = [bitarray(bin(i)[2:]) for i in range(32)]

    bits = [
        bitarray([
            (lambda v: 0 if v < (n - len(bit)) else bit[v - (n - len(bit))])(i)
            for i in range(n)
        ]) for bit in bits
    ]


    for bit in bits:
        print(bit)


    sod = SOD(4)
    print("sod: ", sod)

    print("setting {} and {} to True".format(bits[1], bits[2]))
    sod[bits[1]] = True
    sod[bits[2]] = True

    print("sod: ", sod)

    print("setting {} to True".format(bits[3]))
    sod[bits[3]] = True

    print("sod: ", sod)

    print("setting {} to False".format(bits[4]))
    sod[bits[4]] = False

    print("sod: ", sod)

    print("setting {} to False".format(bits[5]))
    sod[bits[5]] = False

    print("sod: ", sod)

    print("="*100)
    print("=" * 100)

    for bit in bits:
        print("-"*100)
        print("trying to access value of {}".format(bit))
        try:
           print("{}: {}".format(bit, sod[bit]))
        except KeyError:
            print("key not found")

