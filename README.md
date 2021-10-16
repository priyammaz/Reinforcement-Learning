# Project Greed
### Group: LeepDearning
### Members: Priyam Mazumdar, Gowathami Venkateswaran, Yang Yue

## Setup:
###  Setting Up Environment
Install anaconda on your computer and use the following commands to build the environment

```
conda create --name "gym_atari" python=3.8
conda install pip
pip install -r requirements.txt 
```

### Setting up Atari Games:
- Download ROMS from [here](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html).  
- Uncompress RAR file and then unzip Roms file.
- Go to directory with Roms file and use following command

```
ale-import-roms ROMS/ROMS/
```



