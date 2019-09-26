import os
import sys
import time
from datetime import datetime
import gym
import gym_gazebo2
import tensorflow as tf
import multiprocessing

from importlib import import_module
from baselines import bench, logger
from baselines.ddpg import ddpg
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

ncpu = multiprocessing.cpu_count()

if sys.platform == 'darwin':
    ncpu //= 2

config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu,
                        log_device_placement=False)

config.gpu_options.allow_growth = True

tf.Session(config=config).__enter__()

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module

def get_learn_function(alg, submodule=None):
    return get_alg_module(alg, submodule).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def make_env():
    env = gym.make('PhantomX-v0')
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)

    return env

# Get dictionary from baselines/ppo2/defaults
env_type = 'phantomx_mlp'
alg_kwargs = get_learn_function_defaults('ddpg', env_type)
alg_kwargs['env_name'] = 'PhantomX-v0'
# Create needed folders
timedate = datetime.now().strftime('%Y-%m-%d_%Hh%Mmin')
logdir = '/tmp/ros2learn/PhantomX-v0/ddpg/' + timedate

# Generate tensorboard file
format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
logger.configure(os.path.abspath(logdir), format_strs)

with open(logger.get_dir() + "/parameters.txt", 'w') as out:
    pass

env = DummyVecEnv([make_env])

learn = get_learn_function('ddpg')
# transfer_path = alg_kwargs['transfer_path']
transfer_path = None
# Remove unused parameters for training
alg_kwargs.pop('env_name')
alg_kwargs.pop('trained_path')
alg_kwargs.pop('transfer_path')

if transfer_path is not None:
    # Do transfer learning
    #learn(env=env,load_path=transfer_path, **alg_kwargs)
    learn(env=env,load_path=transfer_path, network='mlp')
else:
    #learn(env=env, **alg_kwargs)
    learn(env=env, **alg_kwargs)
