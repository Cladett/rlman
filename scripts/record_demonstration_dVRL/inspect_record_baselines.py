"""
@brief    This script is used to test the recorded demonstration using baselines.
          And check the format of the recorded demonstrations.
          Takes as input .npz file and print screen info
@author   Claudia D'Ettorre (c.dettorre@ucl.ac.uk)
@date     18 Mar 2021
"""

import argparse
import numpy as np

def add_parameters(parser): 
    parser.add_argument("--npz-path", help="Path to where npz file.")

def main():
    # Read command line parameters 
    parser = argparse.ArgumentParser() 
    add_parameters(parser)
    args = parser.parse_args()

    # Inspecting the dataset
    dataset = np.load(args.npz_path, allow_pickle=True)
    print(dataset.files)

    print('Actions shape', dataset['acs'].shape)
    print('Obs shape', dataset['obs'].shape)
    print('Obs shape', dataset['info'].shape)
    print('Actions type', type(dataset['acs']))
    #print(dataset['actions'])
    print('Obs type', type(dataset['obs']))
    return


if __name__=='__main__':
    main()




