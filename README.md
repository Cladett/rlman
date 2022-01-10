# rlman: RL environment for organ manipulation compatible with the dVRK
Reinforcement Learning environments for training surgical sub-tasks using the dVRK.

## Description
Simulation environment for training RL agents using [OpenAI Baselines](https://github.com/openai/baselines)

## Requirements
System set up, note that only linux is supported:

Requires CoppeliaSim: https://coppeliarobotics.com/downloads

Requires NVIDIA Container Runtime for Docker: https://github.com/NVIDIA/nvidia-docker

Enable GUI for Docker containers: http://wiki.ros.org/docker/Tutorials/GUI

## Installation and virtualenvironment 
For sanity perspective, it is recommended to use virtual environments (virtualenvs) to make sure packages from different projects do not interfere with each other. 
To install virtualenv:

`pip install virtualenv`

Virtualenvs are folders that have copies of python executable and all python packages. To create a virtualenv called rlman with python3.7 run:

`virtualenv --python=python3.7 /path/to/rlman`

To activate the environment:

`. /path/to/venv/bin/activate`

One you have the venv activated install the rlman as follows:
```
git clone git@github.com:Cladett/rlman.git
cd rlman
python setup.py install
```

## Usage
CoeppliaSim can be launched within docker in hidden mode. Add the "-h" flag in the final line:

```
CMD $COPPELIASIM_ROOT_DIR/coppeliaSim -s -h -q /opt/scene.ttt 
```

Everytime a modification gets done to the dockerfiles, run the following script to build the images:

```
python build_images.py --input-dir path/to/docker_files
```
