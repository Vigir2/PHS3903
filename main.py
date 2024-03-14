import numpy as np
import init_functions as init

if __name__ == "__main__":
    print("Hello World")
    atoms = ['O', 'H', 'H', 'H']
    with open("h2O.xyz", "w") as file:
        file.write("3\n")
        pos = init_water(3)
        for i in range(len(pos)):
            file.write(f"{atoms[i]}     {pos}")