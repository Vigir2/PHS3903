import numpy as np
import parameters.h2O_model as h2O
from pint import UnitRegistry
from H2O import H2O
import multiprocessing
ureg = UnitRegistry()

def do_something(m: H2O):
    m.O_force = 69
    m.H1_force = 42
    return m
    #print(m.inertia_tensor())

if __name__ == "__main__":
    eps = h2O.epsilon_SI
    #print(eps)
    #print(ureg("kJ/mol").to("J/mol").magnitude)
    water_molecules = [H2O(3, 300) for _ in range(15)]
    pool = multiprocessing.Pool()
    x = pool.map(do_something, water_molecules)
    pool.close()
    pool.join()
    print(x)
    print(x[0].O_force)
    print(water_molecules[0].O_force)
    water_molecules = [H2O(3, 300) for _ in range(15)]
    for m in water_molecules:
        m.O_force = 69
        m.H1_force = 42
    print(water_molecules[0].O_force)

    

"""     print("Hello World")
    atoms = ['O', 'H', 'H', 'H']
    with open("h2O.xyz", "w") as file:
        file.write("3\n")
        pos = init_water(3)
        for i in range(len(pos)):
            file.write(f"{atoms[i]}     {pos}") """