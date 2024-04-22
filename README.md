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

```
python SAC.py
```

##### Edit configuration

```
python SAC.py [-e episodes] [-n noise] [-ef engine_failure] [-bs buffer_size] [-s seed] [-se save_every] [-sn save_name] [-bs batch_size]
```

Description:

- `-e` or `--episodes` (int): Total number of episodes to train for (default: 1000)
- `-n` or `--noise` (float): Sigma value for the Gaussian distribution of noisy observations (default: None)
- -ef or --engine_failure (float): Probability of engine failure (default: None)
- `-bs` or `--bufer_size` (int): Size of the replay buffer (default: 100000)
- `-s` or `--seed` (int): Random Seed (default: 1)
- `-se` or `--save_every` (int): Save the model every n episodes (default: 100)
- `-sn` or `--save_name` (str): The name of the saved model. (defualt "SAC")
- `-bs` or `--batch_size` (int): Size of the training batch (default: 256)

### Results

##### Midterm report results (original model)

![SAC reward](https://github.com/yyanghly/SAC-Discrete/assets/99605351/49575458-e8c4-43d6-912e-a5a31be2c384)

![SAC_policy_loss](https://github.com/yyanghly/SAC-Discrete/assets/99605351/7ce61f87-c043-4b4b-8ef5-ffcba65449f3)

![SAC_bellmann_error](https://github.com/yyanghly/SAC-Discrete/assets/99605351/e4ea364c-a8b3-420f-901c-1bdd9b677aa7)

#### Introducing Uncertainty

##### Original model

![1713763961657](image/README/1713763961657.png)![1713763978888](image/README/1713763978888.png)

##### Introduce Noisy Observation:

![1713763472967](image/README/1713763472967.png)

![1713763496522](image/README/1713763496522.png)

##### Introduce Random Engine Failure

![1713763524630](image/README/1713763524630.png)

![1713763740519](image/README/1713763740519.png)
