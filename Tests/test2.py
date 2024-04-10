import numpy as np

def test(**kwargs):
    print(list(kwargs.values()))
        
    for key, value in kwargs.items():
        print(key, value)
    

test(auto=True, énergie=False)

print(np.array([1,2,4]))