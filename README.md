
__This is a heavily modified fork from AcutronicRobotics.__

<a href="http://www.acutronicrobotics.com"><img src="https://github.com/AcutronicRobotics/gym-gazebo2/blob/master/imgs/alr_logo.png" align="left" width="190"></a>


This repository contains a number of ROS and ROS 2 enabled Artificial Intelligence (AI)
and Reinforcement Learning (RL) [algorithms](algorithms/) that run in selected [environments](environments/).

The repository contains the following:
- [algorithms](algorithms/): techniques used for training and teaching robots.
- [environments](environments/): pre-built environments of interest to train selected robots.
- [experiments](experiments/): experiments and examples of the different utilities that this repository provides.

A whitepaper about this work is available at https://arxiv.org/abs/1903.06282. Please use the following BibTex entry to cite our work:
```
@misc{1903.06282,
Author = {Yue Leire Erro Nuin and Nestor Gonzalez Lopez and Elias Barba Moral and Lander Usategui San Juan and Alejandro Solano Rueda and Víctor Mayoral Vilches and Risto Kojcev},
Title = {ROS2Learn: a reinforcement learning framework for ROS 2},
Year = {2019},
Eprint = {arXiv:1903.06282},
}
```

---

## Installation

Please refer to [Install.md](/Install.md) to install from sources.


___This is not tested and most probably not working with my fork! Working docker example is coming soon!__
~~Refer to [docker/README.md](/docker/README.md) for ROS2Learn Docker container installation and usage instructions.~~

## Usage

### Tune hyperparameters
Check the optimal network hyperparameters for the environment you want to train. [Hyperparams.md](/Hyperparams.md).
__currently no phantomx hyperparameters available__

### Train an agent
You will find all available examples at */experiments/examples/*. Although the algorithms are complex, the way to execute them is really simple. For instance, if you want to train the phantomx robot using ppo2_mlp, you should execute the following command:

```sh
cd ~/ros2learn/experiments/examples/PHANTOMX
python3 train_ppo2_mlp.py
```

Note that you can add the command line arguments provided by the environment, which in this case are provided by the gym-gazebo2 Env. Use `-h` to get all the available commands.

If you want to test your own trained neural networks, or train with different environment form gym-gazebo2, or play with the hyperparameters, you must update the values of the dictionary directly in the corresponding algorithm itself. For this example, we are using *ppo2_mlp* from [baselines](https://github.com/kkonen/ros2learn/tree/master/algorithms) submodule, so you can edit the `phantomx_mlp()` function inside [baselines/ppo2/defaults.py](https://github.com/kkonen/baselines-1/blob/master/baselines/ppo2/defaults.py).

### Run a trained policy
Once you are done with the training, or if you want to test some specific checkpoint of it, you can run that using one of the running-scripts available. This time, to follow with the example, we are going to run a saved ppo2_mlp policy.

First, we will edit the already mentioned `phantomx_mlp()` dictionary, in particular the `trained_path` value, in [baselines/ppo2/defaults.py](https://github.com/kkonen/baselines-1/blob/master/baselines/ppo2/defaults.py) to the checkpoint we want (checkpoints placed by default in /tmp/ros2learn). Now we are ready to launch the script.

Since we want to visualize it in real conditions, we are also going to set some flags:

```sh
cd ~/ros2learn/experiments/examples/PHANTOMX
python3 run_ppo2_mlp.py -g
```

This will launch the simulation with the visual interface.

### Visualize training data on tensorboard

The logdir path will change according to the used environment ID and the used algorithm in training.
Now you just have to execute Tensorboard and open the link it will provide (or localhost:port_number) in your web browser. You will find many useful graphs like the reward (eprewmean) plotted there.

You can also set a specific port number in case you want to visualize more than one tensorboard file from different paths.

```sh
tensorboard --logdir=/tmp/ros2learn/Phantomx-v0/ --port 8008
```
