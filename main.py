import numpy as np
import parameters.h2O_model as h2O
from pint import UnitRegistry
ureg = UnitRegistry()

if __name__ == "__main__":
    eps = h2O.epsilon_SI
    #print(eps)
    #print(ureg("kJ/mol").to("J/mol").magnitude)
    print(eps * ureg("kJ/mol").to("amu*angstrom^2/molecule/fs^2").magnitude)
    print(h2O.epsilon)
    

"""     print("Hello World")
    atoms = ['O', 'H', 'H', 'H']
    with open("h2O.xyz", "w") as file:
        file.write("3\n")
        pos = init_water(3)
        for i in range(len(pos)):
            file.write(f"{atoms[i]}     {pos}") """