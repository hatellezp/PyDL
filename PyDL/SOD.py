from collections import OrderedDict
from bitarray import bitarray

from typing import Union, Optional


class SOD:
    """
    how this works: this is nothing else than a wrapper around an ordered dict
    with some other stuff

    the keys are bitarrays modeling subsets of a bigger set
    the values stored are True or False for consistency or inconsistency
    respectively.

    two things can happen here:
        - suppose you add a new entry with value False, meaning inconsistency
          then, whenever you try to add a new entry modeling a superset of this
          set it will be not necessary, as you will already know the answer
        - suppose you add a new entry with value True, then all of the already
          existent entries which model subsets of this set should be deleted,
          as its consistency it is now implied by the new entry
    """
    def __init__(self, size: int):
        self.size = size
        self.od = OrderedDict()

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, item):
        pass


if __name__ == "__main__":
    print("hello there from the SOD module!")
