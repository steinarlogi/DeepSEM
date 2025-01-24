import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--from_file', type=str, help='deep sem grn file', required=True)
parser.add_argument('--save_dir', type=str, help='The output directory', required=True)
parser.add_argument('--file_name', type=str, help='The name of the file in which to save the output(optional)')

args = parser.parse_args()

save_dir = args.save_dir
grn_file = args.from_file
_, filename = os.path.split(grn_file)
filename, _ = os.path.splitext(filename)
filename = filename + '.csv'

if args.file_name is not None:
    filename = args.file_name.removesuffix('.csv')
    filename = filename + '.csv'

try:
    os.mkdir(save_dir)
except:
    print ('Dir exists')

new_data = []
# Add headers
new_data.append(['Regulator', 'Target', 'Weight', 'Sign'])

with open(grn_file) as f:
    lines = f.readlines()

    for i in range(1, len(lines)):
        line = lines[i].split('\t')
        regulator = line[0].strip(' \n')
        target = line[1].strip(' \n')
        weight = line[2].strip(' \n')
        sign = str(int(np.sign(float(weight))))

        new_line = [regulator, target, weight, sign]

        new_data.append(new_line)

new_data = [','.join(line) + '\n' for line in new_data]

# Write to the new file
new_path = os.path.join(save_dir, filename)
with open(new_path, mode='w') as f:
    f.writelines(new_data)



