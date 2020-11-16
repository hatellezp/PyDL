import ctypes as ct

"""
enum State {
    SUCCESS = 0,
    UNINVERTIBLE = 1,
    UNKOWN = 2,
    EQUIVALENT = 3,
    UNEQUIVALENT = 4,
};
"""


class Result(ct.Structure):
    """
    typedef struct {
        enum State state;
        double number;
    }vgraph_result;
    """
    _fields_ = ("state", ct.c_int), ("value", ct.c_double), ("message", ct.c_char_p),  # maybe this is c_wchar_p we need to test it


def to_double_array(from_list):
    length = len(from_list)
    to_list = (ct.c_double * length)(0)

    for i in range(length):
        to_list[i] = from_list[i]

    return to_list