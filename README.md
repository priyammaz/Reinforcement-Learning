# Project Greed
#### Group: LeepDearning
#### Members: Priyam Mazumdar, Gowathami Venkateswaran, Yang Yue

### Reinforcement Learning
In this project we will explore Q-Learning and its different variants. We will use OpenAI Gym as our environment of 
training. More specifically we will be studying the results of different models and methodologies on _Cartpole-v1_ and
_LunarLander-v2_ from Box2D and _Pong-v0_ from Atari. 

Our focus is on studying DeepQ, Double DeepQ and Dueling DeepQ Learning algorithms to understand how quickly we can solve
the game. Also we will be testing the difference in training performance when using numerical simulated values offered
by the Box2D games versus using an image input. 

## Setup:
###  Setting Up Environment
Install anaconda on your computer and use the following commands to build the environment. This will do the following:
- Create a **conda** environment called gym with Python 3.8
- Install requirements for testing (with the exception of Pytorch)
- Install preloaded games found in /roms folder

```
bash setup.sh
```

### Setting up Pytorch
Go to [here](https://pytorch.org) to get the conda command for your OS and Version

### Setting up Atari Games:
#### The ROMS for two games are [provided](roms/) but to add additional games follow instructions below
- Download ROMS from [here](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html).  
- Uncompress RAR file and then unzip ROMS file.
- Copy all ROMS into the /roms folder
- Use the following command to pull in the supported games

```angular2html
ale-import-roms roms/
```




