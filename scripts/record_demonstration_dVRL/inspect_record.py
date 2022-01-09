"""
@brief    This script is used to test the recorded demonstration.
          And check the format of the recorded demonstrations.
          Takes as input .npz file and print screen info
@author   Claudia D'Ettorre (c.dettorre@ucl.ac.uk)
@date     1 Sep 2020
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

    print('Number of recorded trajectory', dataset['episode_returns'].shape)
    print('Actions shape', dataset['actions'].shape)
    print('Obs shape', dataset['obs'].shape)
    print('Rewards shape', dataset['rewards'].shape)
    print('Episode starts shape', dataset['episode_starts'].shape)
    print('Episode returns shape', dataset['episode_returns'].shape)
    print('Actions type', type(dataset['actions']))
    #print(dataset['actions'])
    print('Obs type', type(dataset['obs']))
    print('Rewards type', type(dataset['rewards']))
    print('Episode returns type', type(dataset['episode_returns']))
    print('Episode starts type', type(dataset['episode_starts']))
    return


if __name__=='__main__':
    main()




