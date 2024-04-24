
init_universe = "System \"{name}\" has been initialized succesfully with {N} water molecules at {T:.2f} K and {P:.2f} bar."

init_error = "An error has occured while initiating system \"{name}\":\n"

error_water_placement = "Could not place {N} water molecules in a {a} x {a} x {a} cell with a minimum distance of {security_distance} Angstrom.\nTry to increase cell size or decrease the number of molecule."

write_file = "File {fname} has been written succesfully!"

write_state_variables = "System state variables for {var} have been written in {loc} sucesfully!"

invalid_input = "{input} input is invalid. {input} must be a {type}."

nve_initiation = "NVE velocity Verlet scheme initiated for {time} ps with {n} iterations."

npt_initiation = "NPT velocity Verlet scheme initiated for {time} ps with {n} iterations. Temperature and pressure targets are {T:.2f} K and {P:.2f} bar."

error_temperature = "An error occured while computing system temperature:\nToo many arguments provided : {a}"