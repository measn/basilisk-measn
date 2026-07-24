import os
import glob
import numpy as np
import h5py
import pyvista as pv
import xml.etree.ElementTree as ET

def convert_pvtu_to_xdmf(input_dir="vtu", output_name="simulation_results"):
    """
    Safely converts Basilisk parallel VTU files into a unified HDF5 dataset 
    with an associated XDMF3 metadata wrapper for ParaView visualization.
    Dynamically resolves scalar, vector, and tensor dimensions to prevent
    HDF5 hyperslab reading errors.
    """
    search_pattern = os.path.join(input_dir, "*.pvtu")
    pvtu_files = sorted(glob.glob(search_pattern))
    
    if not pvtu_files:
        print(f"Error: No .pvtu files found in '{input_dir}' directory.")
        return

    h5_filename = f"{output_name}.h5"
    xmf_filename = f"{output_name}.xmf"

    # Initialize XDMF XML tree structure for XDMF3 temporal collection
    xdmf = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(xdmf, "Domain")
    grid_collection = ET.SubElement(domain, "Grid", GridType="Collection", CollectionType="Temporal")

    print(f"Processing {len(pvtu_files)} timesteps into {h5_filename} and {xmf_filename}...")

    # Open HDF5 file in write mode
    with h5py.File(h5_filename, "w") as h5f:
        for step_idx, pvtu_path in enumerate(pvtu_files):
            filename = os.path.basename(pvtu_path)
            
            # Extract physical time from filename (expected format: fields_t_0.0050.pvtu)
            try:
                time_val = float(filename.split("_t_")[1].split(".pvtu")[0])
            except (IndexError, ValueError):
                time_val = float(step_idx)

            print(f"Converting step {step_idx}: t = {time_val}")

            try:
                # Read parallel dataset using PyVista XML reader to resolve vtk_pieces
                reader = pv.XMLPUnstructuredGridReader(pvtu_path)
                dataset = reader.read()
                
                # Extract point coordinates (enforce 64-bit float precision)
                points = dataset.points.astype(np.float64)
                num_points = points.shape[0]
                
                # Safely parse cell connectivity from PyVista unstructured grid
                # VTK strictly requires 32-bit signed integers (int32) for topology mapping
                cells = dataset.cells
                connectivity_list = []
                cell_count = 0
                i = 0
                
                while i < len(cells):
                    n_pts = cells[i]
                    connectivity_list.append(cells[i + 1 : i + 1 + n_pts])
                    cell_count += 1
                    i += 1 + n_pts
                    
                connectivity = np.array(connectivity_list, dtype=np.int32)
                num_cells = cell_count

                # Create HDF5 group for the current timestep
                group = h5f.create_group(f"Step_{step_idx}")
                
                # Write geometry and topology using gzip compression
                group.create_dataset("Points", data=points, compression="gzip")
                group.create_dataset("Connectivity", data=connectivity, compression="gzip")
                
                # Store all cell-centered fields natively present in the VTU
                cell_data_group = group.create_group("CellData")
                for array_name in dataset.cell_data.keys():
                    field_data = np.ascontiguousarray(dataset.cell_data[array_name], dtype=np.float64)
                    cell_data_group.create_dataset(array_name, data=field_data, compression="gzip")

                # Build XDMF XML nodes for the temporal collection
                grid = ET.SubElement(grid_collection, "Grid", Name=f"Mesh_t_{time_val}", GridType="Uniform")
                ET.SubElement(grid, "Time", Value=str(time_val))
                
                # Topology definition (Quadrilateral elements, int32 precision = 4 bytes)
                topology = ET.SubElement(grid, "Topology", TopologyType="Quadrilateral", NumberOfElements=str(num_cells))
                dataitem_top = ET.SubElement(topology, "DataItem", Dimensions=f"{num_cells} 4", NumberType="Int", Precision="4", Format="HDF")
                dataitem_top.text = f"{h5_filename}:/Step_{step_idx}/Connectivity"
                
                # Geometry definition (XYZ coordinates, float64 precision = 8 bytes)
                geometry = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
                dataitem_geo = ET.SubElement(geometry, "DataItem", Dimensions=f"{num_points} 3", NumberType="Float", Precision="8", Format="HDF")
                dataitem_geo.text = f"{h5_filename}:/Step_{step_idx}/Points"
                
                # Dynamic Attribute fields mapping (Handles Scalars, Vectors, and Tensors)
                for array_name in dataset.cell_data.keys():
                    field_data = np.ascontiguousarray(dataset.cell_data[array_name], dtype=np.float64)
                    shape = field_data.shape
                    
                    # Evaluate array dimensions to define appropriate XDMF format
                    if len(shape) == 1:
                        attr_type = "Scalar"
                        dims_str = str(shape[0])
                    elif len(shape) == 2 and shape[1] == 3:
                        attr_type = "Vector"
                        dims_str = f"{shape[0]} 3"
                    elif len(shape) == 2 and shape[1] == 9:
                        attr_type = "Tensor"
                        dims_str = f"{shape[0]} 9"
                    else:
                        attr_type = "Matrix"
                        dims_str = " ".join(map(str, shape))

                    attribute = ET.SubElement(grid, "Attribute", Name=array_name, AttributeType=attr_type, Center="Cell")
                    dataitem_attr = ET.SubElement(attribute, "DataItem", Dimensions=dims_str, NumberType="Float", Precision="8", Format="HDF")
                    dataitem_attr.text = f"{h5_filename}:/Step_{step_idx}/CellData/{array_name}"

            except Exception as e:
                print(f"Warning: Failed to process {filename}. Reason: {e}")

    # Write out the formatted XDMF wrapper file
    tree = ET.ElementTree(xdmf)
    ET.indent(tree, space="  ")
    tree.write(xmf_filename, encoding="utf-8", xml_declaration=True)
    
    print(f"\nConversion successfully completed.")
    print(f"-> Open '{xmf_filename}' in ParaView using 'Xdmf3 Reader T'.")

if __name__ == "__main__":
    convert_pvtu_to_xdmf()