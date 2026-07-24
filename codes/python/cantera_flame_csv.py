import cantera as ct
import pandas as pd
import numpy as np

mech_path = '/home/measn/Documents/Cantera/python/database/C3MechV4_RED/C3MechV4_RED.yaml' 
gas = ct.Solution(mech_path)

# operating conditions
T0 = 300.0          
p0 = ct.one_atm     
phi = 1.0           
fuel = 'IC3H7OH'
oxidizer = 'O2:0.21, N2:0.79'

gas.TP = T0, p0
gas.set_equivalence_ratio(phi, fuel, oxidizer)
    
# 1D flame
width = 0.05  # Domain width [m]
flame = ct.FreeFlame(gas, width=width)
flame.set_refine_criteria(ratio=3.0, slope=0.05, curve=0.05)
flame.solve(loglevel=1, auto=True)


# export
data = pd.DataFrame()
data['x'] = flame.grid              
data['u'] = flame.velocity         
data['T'] = flame.T            
data['rho'] = flame.density      

# Extract all species mass fractions
for i, species in enumerate(gas.species_names):
    data[f'Y_{species}'] = flame.Y[i]

grad_T = np.gradient(flame.T, flame.grid)
x_flame_front = flame.grid[np.argmax(grad_T)]
data['x'] = data['x'] - x_flame_front

# csv
output_filename = 'flame_profile_isoprop.csv'
data.to_csv(output_filename, index=False)

print(f"youpla boom c'est bon tout roule")
print(f"file name : {output_filename}.")