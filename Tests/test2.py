

#!/usr/bin/env python3
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
import numpy as np

def func(a, b):
    return a + b, a

def main():
    a_args = [1,2,3]
    second_arg = 1
    with Pool() as pool:
        L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
        M = pool.starmap(func, zip(a_args, repeat(second_arg)))
        N = pool.map(partial(func, b=second_arg), a_args)
        print(L)
        L = np.array(L)
        print(L[:,0])
        print(L[:,1])


if __name__=="__main__":
    # freeze_support()
    # main()
    a = 6
    b = 9
    print(f"{a + b = }")