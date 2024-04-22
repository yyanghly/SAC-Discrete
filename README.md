# SAC-Discrete - COMP3340 Group Project

## Overview

This GitHub repository contains the work for the COMP3340 course's group project. This repo focuses on the implementation of Discrete Soft Actor-Critic algorithm.

### Installation

```
sudo apt install -y python3-opengl xvfb swig
pip install -r requirements.txt
```

### Run

Below is the command to run the algorithm. The trainings are logged with wandb.ai.

##### Run

```
python SAC.py [-e episodes] [-n noise] [-ef engine_failure] [-bs buffer_size] [-s seed] [-se save_every] [-sn save_name] [-bs batch_size]
```

Description:

- `-e` or `--episodes` (int): Total number of episodes to train for (default: 1000)
- `-n` or `--noise` (float): Sigma value for the Gaussian distribution of noisy observations (default: None)
- `-ef` or `--engine_failure` (float): Probability of engine failure (default: None)
- `-bs` or `--bufer_size` (int): Size of the replay buffer (default: 100000)
- `-s` or `--seed` (int): Random Seed (default: 1)
- `-se` or `--save_every` (int): Save the model every n episodes (default: 100)
- `-sn` or `--save_name` (str): The name of the saved model. (defualt "SAC")
- `-bsz` or `--batch_size` (int): Size of the training batch (default: 256)

Specifically, the script used in midterm report is:

```
python SAC.py -n=2000
```

And the script used in final presentation to introduce noisy observation is

```
python SAC.py -n=10000 -noise=0.01 -sn="SAC_noise_0.01"
```

THe script used in final presentation to introduce random engine failure is:

```
python SAC.py -n=10000 -ef=0.1 -sn="SAC_engine_failure_0.1"
```

### Results

#### Original model (used in midterm report)

2000 episodes:

<p float="left">
  <img src="image/README/1713765035533.png" width="48%" />
  <img src="image/README/1713765051074.png" width="48%" /> 
</p>

10000 episodes:

<p float="left">
  <img src="image/README/1713763961657.png" width="48%" />
  <img src="image/README/1713763978888.png" width="48%" /> 
</p>

#### Introducing Uncertainty (used in final presentation)

##### Introduce Noisy Observation:

<p float="left">
  <img src="image/README/1713763472967.png" width="48%" />
  <img src="image/README/1713763496522.png" width="48%" /> 
</p>

##### Introduce Random Engine Failure:

<p float="left">
  <img src="image/README/1713763524630.png" width="48%" />
  <img src="image/README/1713763740519.png" width="48%" /> 
</p>
