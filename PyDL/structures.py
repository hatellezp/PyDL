################################################################################
################################################################################
from typing import Tuple, Iterable, Union, Generator, Callable, Optional
from collections import OrderedDict
import numpy as np
import math

from bitarray import bitarray

from utils import ncr, not_empty_subsets, banker_sequence
from SOD import SOD


################################################################################
################################################################################
# TODO: decide if I need a relation class or not

################################################################################
################################################################################


class Element:
    """
    this class packages an element with three attributes
    the relation name, the element name and its value, that is
    an element is of the form R(x) with value v
    """

    def __init__(self, _rname: str, _ename: str, _cred: Optional[float], _neg: bool = True):
        self.rname = _rname
        self.ename = _ename
        self.neg = _neg
        self.evalue: Union[None, float] = None
        self.cred = 1. if _cred is None else _cred

    def __eq__(self, other):
        if not isinstance(other, Element):
            return False
        else:
            return self.rname == other.rname and self.ename == other.ename

    def r_name(self) -> str:
        return self.rname

    def e_name(self) -> str:
        return self.ename

    def e_value(self) -> Union[None, float]:
        return self.evalue

    def mod_evalue(self, v: float) -> None:
        if self.evalue is None:
            self.evalue = v

    def negate(self) -> None:
        self.neg = not self.neg

    def copy(self) -> 'Element':
        return Element(_rname=self.rname, _ename=self.ename, _cred=self.cred, _neg=self.neg)


class Bag:
    """
    for the moment I'm implementing the Bag as immutable
    We will see after if this is good or not
    """

    def __init__(self, _data: Iterable[Element]):

        self.data = OrderedDict()
        accum = 0
        for e in _data:
            self.data[accum] = e
            accum += 1

        self.size = accum

        # if no conf limit is provided then put the limit as the size of the
        # set is set as the size limit

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, key: int) -> Element:
        if key > self.size:
            raise IndexError("Bag: index out of bounds.")  # good error type
        else:
            return self.data[key]

    def __setitem__(self, key: int, e: Element) -> None:
        if key > self.size:
            raise IndexError("Bag: index out of bounds.")
        else:
            self.data[key] = e

    def __contains__(self, item: Element) -> bool:
        return item in self.data.values()

    def add(self, e: Element) -> bool:
        if e in self:  # now it works, thanks to __contains__
            return False
        else:
            try:
                self.data[self.size + 1] = e
                self.size += 1
                return True
            except:  # a better exception here ?
                print("Couldn't add new element...")
                return False

    # TODO: verify this
    def __iter__(self):
        if self.size == 1:
            return None
        else:
            self.x = 0
            return self  # I'm not sure about this

    def __next__(self):
        x = self.x
        if x > self.size:
            raise StopIteration
        else:
            self.x += 1

        return self.data[x]

    @staticmethod
    def subbag(b: 'Bag', new_filter: Union[np.ndarray, bitarray],
               old_sb: Optional['Bag'] = None,
               old_filter: Optional[Union[np.ndarray, bitarray]] = None) -> 'Bag':

        if old_filter is None or old_sb is None:
            return Bag([b[i] for i in new_filter if i])
        else:
            n = b.size
            difference = [old_filter[i] - new_filter[i] for i in np.range(n)]

            # what to do here ???

            pass

    @staticmethod
    def is_subbag(b1: 'Bag', b2: 'Bag') -> bool:
        """
        this method supposed that both bags comes from a same super bag
        and thus the elements are supposed to be ordered in the same fashion

        :param b1:
        :param b2:
        :return:
        """

        # TODO: what follows is a naive method,
        #       find a way to exploit the ordering
        if len(b1) > len(b2):
            return False

        for i in range(len(b1)):
            e1 = b1[i]
            if e1 not in b2:
                return False
        return True


class Rule:
    def __init__(self, _arity: int, _rule: Callable[[Bag], bool]):
        self.arity = _arity
        self.rule = _rule

    def apply(self, b: Bag) -> bool:
        if len(b) == self.arity:  # only apply if correct arity
            return self.rule(b)
        else:
            raise Exception("incorrect arity")

    def get_arity(self) -> int:
        return self.arity


class Reasoner:
    def __init__(self, _rules: Iterable[Rule],
                 _max_inconsistency: Optional[int],
                 _engine: Callable[[Iterable[Rule], Bag], bool],
                 _credibility: Callable[[Bag], float]):  # or int here at the end?
        self.rules = OrderedDict()
        self.max_arity = None
        self.min_arity = None
        self.credibility = _credibility

        accum = 0
        for r in _rules:

            if self.max is None:
                self.max = r.get_arity()
            if self.min is None:
                self.min = r.get_arity()

            self.min_arity = min(self.min_arity, r.get_arity())
            self.max_arity = max(self.max_arity, r.get_arity())

            self.rules[accum] = r
            accum += 1

        self.size = accum
        self.max_inconsistency = _max_inconsistency  # It may depend of the underlying logic or of the current rules themselves
        self.engine = _engine

    def compile_conflict_matrix_detailed(self, b: Bag) -> SOD:
        """

        :param b: a bag of facts
        :return: a SOD (subset ordered dict
        """
        # size of the bag
        b_lenght = len(b)

        # create an SOD
        sod = SOD(size=b_lenght)

        conflict_size_limit = self.max_inconsistency if (self.max_inconsistency
                                                         is not None and self.max_inconsistency < b_lenght) \
            else b_lenght

        """
        two things:
            - create generators for the subsets until reach of 'conflict_size_limit'
            - populate the SOD 
        """

        for subset_size in range(1, conflict_size_limit + 1):
            subset_generator = banker_sequence(b_lenght, subset_size)

            # for the moment I don't have an idea to update the bags
            # Okay now I have an idea:
            # try to search for a value, and only if it doesn't exist, then
            # call the engine
            for subset in subset_generator:
                subbag = Bag.subbag(b, subset)

                # try to get the value, if it doesn't exist, only then call the
                # engine, which is a priori more expensive in computing power
                try:
                    a = sod[subset]
                except KeyError:
                    sod[subset] = self.engine(self.rules.values(), subbag)

        # as a bonus, all subsets that have sod[subset] as False are minimal :)
        # that's it
        return sod

    def compile_conflict_matrix_compressed(self, b: Bag, sod: SOD = None) -> np.ndarray:
        # compute the sod if it doesn't exist
        sod = sod if sod is not None else self.compile_conflict_matrix_detailed(self, b)
        pass

        b_lenght = len(b)

        conflict_size_limit = self.max_inconsistency if (self.max_inconsistency
                                                         is not None and self.max_inconsistency < b_lenght) \
            else b_lenght

        # now what ?

    def rank(self, b: Union[Bag, Tuple[Bag, np.ndarray]]) -> None:
        pass


class PyDL:
    def __init__(self, ):
        pass
