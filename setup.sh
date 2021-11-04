#!/bin/bash
conda create --name "gym" python=3.8 # create conda environments
conda activate "gym"
conda install pip # install pip
pip install -r "requirements.txt" # install packages
ale-import-roms roms/ # install games

