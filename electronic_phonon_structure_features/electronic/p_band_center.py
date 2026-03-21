import os
import re
import subprocess

# Non_Metals
metals = [
    "H", "He", "B", "C", "N", "O", "F", "Ne", "P", "S" ,
    "Cl", "Ar", "Se", "Br", "Kr", "I", "Xe", "At", "Rn", "Ts", "Og",
    "B", "Si", "Ge", "As", "Sb", "Te", "Po", "At"
]

# Get the current working directory
cwd = os.getcwd()

# Extract the names of all folders in the current directory
folders = [f.name for f in os.scandir(cwd) if f.is_dir()]

# Function to extract element names from a folder name
def extract_element_names(folder_name):
    return re.findall(r'([A-Z][a-z]*)', folder_name)

# Function to determine whether an element is a metal
def is_metal(element):
    return element in metals

for folder in folders:
    print(f"正在处理 {folder} ...")
    element_names = extract_element_names(folder)
    metal_elements = [name for name in element_names if is_metal(name)]
    
    if metal_elements:
        # Convert metal_elements list to a space separated string
        metal_string = " ".join(metal_elements)
        # Create the string that will be sent to vaspkit's input
        vaspkit_input = f"5\n503\nN\n1\n{metal_string}\n"
        # Run vaspkit with the required input
        try:
            with open(os.devnull, 'w') as fp:
                subprocess.run('vaspkit', input=vaspkit_input, text=True, stdout=fp, stderr=fp, shell=True, cwd=folder)  # Redirect output to /dev/null
            with open(os.path.join(cwd, 'p_band_center.txt'), 'a') as f:  # Use the full path to the file
                f.write(f"{folder}\n")
            
            # Extract data from BAND_CENTER
            with open(os.path.join(folder, 'BAND_CENTER'), 'r') as f:
                content = f.read()
                spin_data = re.findall(r'(Spin-Channel|Spin-UP|Spin-DW)\s+\S+\s+(\S+)', content)
                
            # Write the extracted data to the file
            with open(os.path.join(cwd, 'p_band_center.txt'), 'a') as f:  # Use the full path to the file
                for spin, value in spin_data:
                    f.write(f"{spin}: {value}\n")

            # Rename BAND_CENTER file
            os.rename(os.path.join(folder, 'BAND_CENTER'), os.path.join(folder, 'BAND_CENTER_p'))  # add the directory to the file path

            print(f"{folder} 处理完成")
        except Exception as e:
            print(f"Error processing {folder}: {str(e)}")
            with open(os.path.join(cwd, 'error_p_band_center.txt'), 'a') as f:
                f.write(f"{folder}: {str(e)}\n")

print("所有文件夹处理完成")

