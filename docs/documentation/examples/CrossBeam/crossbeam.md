# Crossbeam test case with FENIAX

This notebook takes you through the steps of using FENIAX to simulation a flexible crossbeam. The test case is taken from Hilger (2024), [_A Dynamic Nonlinear Inertia Relief Formulation_](https://doi.org/10.2514/6.2024-2258). It involves a crossbeam

## Load modules
Here we load the relevant modules used in the simulation.

``` python
import feniax.preprocessor.configuration as configuration  
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax.numpy as jnp
import pathlib
import numpy as np
```

## Housekeeping
We then define a few useful variables. 

``` python
gravity_forces = False
gravity_label = "g" if gravity_forces else ""
label = 'm1'
label_name = label + gravity_label
```

## Case setup
To launch a fines a FENIAX input object for an intrinsic modal dynamic simulation, including the finite-element model files, modal settings, solver configuration, and loading conditions. It applies gravity and a sinusoidal time-dependent point load, then builds the configuration object and runs the simulation to generate the solution.

``` python
inp = Inputs()
# inp.log.level="debug"
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'rbeam': None, 'lbeam': None, 'ubeam': None, 'dbeam': None}
inp.fem.Ka_name = f"./FEM/Ka_{label}.npy"
inp.fem.Ma_name = f"./FEM/Ma_{label}.npy"
inp.fem.eig_names = [f"./FEM/eigenvals_{label}.npy",
                     f"./FEM/eigenvecs_{label}.npy"]
inp.fem.grid = f"./FEM/structuralGrid_{label}"
inp.fem.num_modes = 60
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./results_{label_name}")

inp.simulation.typeof = "single"
# inp.system.name = "s1"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.bc1 = 'free'
inp.systems.sett.s1.xloads.gravity_forces = gravity_forces
inp.systems.sett.s1.t1 = 0.1*10
inp.systems.sett.s1.dt = 0.0001
# inp.system.tn = 100 * 10 + 1
inp.systems.sett.s1.solver_library = "diffrax" #
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")
inp.systems.sett.s1.xloads.dead_forces = True
inp.systems.sett.s1.xloads.dead_points = [[0, 2]]

t = np.linspace(0.0, 1.0, 10001)
freq = 2
amp = 0.1
f = amp * np.sin(2*np.pi*freq*t)
inp.systems.sett.s1.xloads.x = t.tolist()
inp.systems.sett.s1.xloads.dead_interpolation = [f.tolist()]

config =  configuration.Config(inp)
sol = feniax.feniax_main.main(input_obj=config)
