import os

def generate_urdf(obj_path, output_dir):
    # Extract the subfolder name and obj filename
    parts = obj_path.split(os.sep)
    subfolder = parts[-2]  # Assuming 'fabrica' is one level up
    obj_name = os.path.splitext(parts[-1])[0]  # Remove the .obj extension
    
    # Construct the URDF XML
    urdf_content = f'''<?xml version="1.0"?>
<robot name="{obj_name}">
    <link name="{obj_name}">
        <visual>
            <geometry>
                <mesh filename="../../../mesh/fabrica/{subfolder}/{parts[-1]}"/>
            </geometry>
        </visual>
        <collision>
            <geometry>
                <mesh filename="../../../mesh/fabrica/{subfolder}/{parts[-1]}"/>
            </geometry>
            <sdf resolution="512"/>
        </collision>
    </link>
</robot>
'''
    
    # Write the URDF content to a file
    urdf_file_name = f"{obj_name}.urdf"
    urdf_file_path = os.path.join(output_dir, subfolder, urdf_file_name)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(urdf_file_path), exist_ok=True)
    
    with open(urdf_file_path, 'w') as urdf_file:
        urdf_file.write(urdf_content)

# Path to the 'fabrica' folder containing subfolders with .obj files
base_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'fabrica', 'mesh', 'fabrica')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'fabrica', 'urdf', 'fabrica')  # Where URDF files will be saved

# Walk through the directory structure
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith('.obj'):
            obj_file_path = os.path.join(root, file)
            generate_urdf(obj_file_path, output_dir)

print("URDF generation completed.")

