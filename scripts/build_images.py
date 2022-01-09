"""
@brief   Script used for building docker images from docker files.
         It looks inside the input_dir to see which dockerfiles can be built
         and if they exist it builds an image with the same tag of the folder
         name.
@author  Claudia D'Ettorre (c.dettorre@ucl.ac.uk).
@date    27 Nov 2020.
"""
import docker
import argparse
import os 
import docker


def parse_cmdline_args():
    parser = argparse.ArgumentParser(description='Build docker images.')
    parser.add_argument('--input-dir', required=True, 
                        help='Path to the directory with Docker files.')
    args = parser.parse_args()
    return args


def main():
    # Read command line parameters
    args = parse_cmdline_args()

    # List the docker images
    ls = os.listdir(args.input_dir)

    # Build docker images
    client = docker.from_env()
    for dim in ls:
        full_path = os.path.join(args.input_dir, dim)
        try:
            client.images.build(path=full_path, tag=dim.lower())
        except docker.errors.APIError: 
            print('The Dockerfile of the environment ' + dim + ' is missing.')

    return 


if __name__ == '__main__':
    main()
