def write_pdb_file(output_file, atoms):
    with open(output_file, 'w') as f:
        f.write("HEADER    Sample PDB File\n")
        atom_serial = 1
        for atom in atoms:
            f.write(f"ATOM  {atom_serial:5d} {atom['atom_name']:4s} MOL     1    {atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}  1.00  0.00           {atoms[at]}\n")
            atom_serial += 1
        f.write("END\n")