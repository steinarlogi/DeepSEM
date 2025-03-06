import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser('Invert the edges from grn_benchmark')
parser.add_argument('--input_dir', required=True, help='The directory containing the original edge lists', type=str)

args = parser.parse_args()

out_dir = os.path.join(args.input_dir, 'inverted')

try:
    os.makedirs(out_dir)
except:
    print (f'Directory {out_dir} exists')


counter = 0

for file in os.listdir(args.input_dir):
    if file.endswith('.csv'):
        data = pd.read_csv(os.path.join(args.input_dir, file))
        # Swap the columns
        original_columns = list(data.columns).copy()
        i = list(data.columns)
        a, b = i.index('Regulator'), i.index('Target')
        i[b], i[a] = i[a], i[b]
        data = data[i]
        data.columns = original_columns

        data.to_csv(os.path.join(out_dir, file), index=False)
        counter += 1

print (f'Finished, {counter} files inverted')

