import numpy as np
import matplotlib.pyplot as plt
# Check the different datas

grn_benchmark_data = []
with open('/home/steinar/Documents/DeepSEM/grn_benchmark_data/GeneNetWeaver_HighNoise_Network1_GeneExpression.csv') as f:
    lines = f.readlines()

    for i in range(1, len(lines)):
        line = lines[i].strip().strip('\n').split(',')
        
        grn_benchmark_data.append(line[1:])

simulated_data = []
with open('/home/steinar/Documents/DeepSEM/simulated_data/data.csv') as f:
    lines = f.readlines()

    for i in range(1, len(lines)):
        line = lines[i].strip().strip('\n').split(',')

        simulated_data.append(line[1:])


# Transpose so that we get an array of size cells x genes
grn_benchmark_data = list(map(list, zip(*grn_benchmark_data)))
simulated_data = list(map(list, zip(*simulated_data)))

grn_benchmark_data = np.array(grn_benchmark_data, dtype=float)
simulated_data = np.array(simulated_data, dtype=float)

print (f'grn_benchmark data shape: {grn_benchmark_data.shape}')
print (f'simulated_data shape: {simulated_data.shape}')

print (f'mean of grn_benchmark data: {np.mean(grn_benchmark_data, axis=1)}')
print (f'mean of simulated data: {np.mean(simulated_data, axis=1)}')

print (f'std of grn_benchmark data: {(np.std(grn_benchmark_data, axis=1))}')
print (f'std of simulated data: {np.std(simulated_data, axis=1)}')
print (f'std of log2 normalized simulated data {np.std(np.log2(simulated_data), axis=1)}')
print (f'std of exp2 normalized simulated data {np.std(np.exp2(simulated_data), axis=1)}')

# Plotting the data to get a better feel for it
bins = 1000
fig, ax = plt.subplots(2, 2)

ax[0, 0].hist(simulated_data.flatten(), bins=bins, alpha=0.9, label='simulated data', density=True)
ax[0, 0].set_title('simulated data')
ax[0, 1].hist(grn_benchmark_data.flatten(), bins=bins, alpha=0.9, label='grn_benchmark data', density=True)
ax[0, 1].set_title('grn_benchmark data')
ax[1, 0].hist(np.log2(simulated_data.flatten()), bins=bins, alpha=0.9, label='log2 normalized simulated data', density=True)
ax[1, 0].set_title('log2 normalized simulated data')
ax[1, 1].hist(np.exp2(grn_benchmark_data.flatten()), bins=bins, alpha=0.9, label='exp2 normalized grn_benchmark data', density=True)
ax[1, 1].set_title('exp2 normalized grn_benchmark data')


plt.show()

