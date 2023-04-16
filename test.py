import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset, load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/tsp", help="Create datasets in data_dir/problem (default 'data/tsp')")
    parser.add_argument("--filename", default="tsp20_test_seed2.pkl", help="Filename of the dataset need to load (ignores datadirs)")
    
    opts = parser.parse_args()
    filename = os.path.join(opts.data_dir, opts.filename)
    
    dataset = load_dataset(filename)

    print(np.shape(dataset))