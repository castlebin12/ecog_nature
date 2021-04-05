import numpy as np
from scipy import io
data = io.loadmat('./S5_raw_segmented.mat')
print('key : ', data.keys())
print('size of data :  ', len(data))
#print('data["data"] : ', data['data'])
for each_column in data:
    print('each columns name : ', each_column, ', size : ', len(data[each_column]))

np_data = np.array(data['data']) # For converting to a NumPy
print(type(np_data))

for np_column in np_data[0][0]:
    print('each np_columns size : ', len(np_column))
print(np_data[0][0])
