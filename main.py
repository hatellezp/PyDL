from PyDL.core_structures import *

core = ct.CDLL("libpydl_core.so")

if __name__=="__main__":
    core.say_hello()

    li = [1,2]
    toli = to_double_array(li)

    print(toli)

    for i in range(len(li)):
        print(toli[i])