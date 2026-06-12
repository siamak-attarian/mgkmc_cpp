import damask
import shutil
# 0. keeping the original
shutil.copy('heteroMap_load_material.hdf5', 'heteroMap_load_material_original.hdf5')
# 1. Load the results file
r = damask.Result('heteroMap_load_material.hdf5')

# 2. Add the derived fields you want
# 'P' (Piola-Kirchhoff stress) and 'F' (Deformation Gradient) are usually there by default.
# We use them to calculate Cauchy stress (sigma) and logarithmic strain (epsilon_V^ln).
r.add_stress_Cauchy()
r.add_strain()
r.add_equivalent_Mises('sigma') # Optional: Adds Mises stress for easy checking

# 3. Export to VTK for ParaView
# This will create files like 'heteroMap_3x3_E_load_material_inc0100.vti'
r.export_VTK()


# import damask

# # Load the results file
# r = damask.Result('heteroMap_3x3_E_load_material.hdf5')

# # Print all available raw fields in the HDF5 file
# print(r.fields)


# import damask

# # 1. Load the results file
# r = damask.Result('heteroMap_3x3_E_load_material.hdf5')

# # 2. Select the 'mechanical' group to make its contents available
# r.select(path='mechanical')

# # 3. Print the fields again to confirm they are loaded
# print(r.fields)