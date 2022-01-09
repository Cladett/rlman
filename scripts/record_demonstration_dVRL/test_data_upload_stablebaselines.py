"""
@brief    Script to test the upload of expert demonstratios as done in 
          stable-baselines
          Give as input parameter the npz file with demonstrations 
@author   Claudia D'Ettorre
@date     28 Aug 2020
"""

import argparse
import gym

# My import
from stable_baselines.gail import ExpertDataset 

def add_parameters(parser): 
    parser.add_argument("--npz-path", help="Path to where npz file.")

def main():
    # Read command line parameters 
    parser = argparse.ArgumentParser() 
    add_parameters(parser)
    args = parser.parse_args()

    # Test file loading
    dataset = ExpertDataset(expert_path=args.npz_path, traj_limitation=-1, 
            batch_size=128, verbose=1)
    print('All done')

    return

if __name__ == '__main__':
    main() 
