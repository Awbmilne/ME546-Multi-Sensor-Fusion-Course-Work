import os
import json
import scipy.io

# Specify where the data root
root_folder = './Data/'

# Specify the list of folder paths
folders = [
    'Calibrations/Long',
    'Calibrations/Short',
    'TrainingData',
    'TestData',
]

# Data object
data = {}

# Nested set
def setval(data, keys, val) -> None:
    lastkey = keys[-1]
    for k in keys[:-1]:  # when assigning drill down to *second* last key
        if k not in data:
            data[k] = {}
        data = data[k]
    data[lastkey] = val

# For each folder path
for folder in folders:
    # Create a list of all files in the folder
    files = os.listdir(root_folder + folder)

    # Loop through all files in the folder
    for file in files:
        # Load the data from the file
        file_data = scipy.io.loadmat(root_folder + folder + '/' + file)
        
        # Use folder name as key set
        keys = folder.split('/') + [file.replace('.mat', '')]
        time_keys = keys + ['time']
        data_keys = keys + ['data']
        
        # Get dimensions of ndarray for data
        raw_data = []
        dims = file_data['data'].shape
        if dims[1] == 1:
            # Convert to basic array
            raw_data = file_data['data'].flatten().tolist()
        else:
            # Convert to list of arrays
            for i in range(dims[1]):
                raw_data.append(file_data['data'][:,i].flatten().tolist())
        
        
        # Write the 'time' and 'data' entries to the data object
        setval(data, time_keys, file_data['time'].flatten().tolist())
        setval(data, data_keys, raw_data)

with open('data.json', 'w') as f:
    json.dump(data, f)
