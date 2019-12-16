import os
import sys
import time
import gym
import gym_gazebo2
import numpy as np
import multiprocessing
import tensorflow as tf

import threading

from importlib import import_module
from baselines import bench, logger
from baselines.ppo2 import model as ppo2
from baselines.ppo2 import model as ppo
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.policies import build_policy

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

def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs

def constfn(val):
    def f(_):
        return val
    return f

def make_env():
    env = gym.make(defaults['env_name'])
    env.set_episode_size(defaults['nsteps'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)

    return env


def leg_obs(main_obs, leg_name):
    obs = 0
    if leg_name is 'lf':
        # obs = np.delete(main_obs, np.s_[108:162])   MAKE THIS SMARTER! NUM_OF_VARIABLES PER LEG * LIMBS PER LEG!
        # positions
        obs = np.delete(main_obs, np.s_[18:27])
        obs = np.delete(obs, np.s_[27:45])
        # ground contacts
        obs = np.delete(obs, np.s_[29])
        obs = np.delete(obs, np.s_[30:32])
        # actions
        obs = np.delete(obs, np.s_[42:45])
        obs = np.delete(obs, np.s_[45:51])
        obs = np.delete(obs, np.s_[36:39])
        # linear velocities
        #obs = np.delete(obs, np.s_[72:90])
        #obs = np.delete(obs, np.s_[54:63])

    if leg_name is 'lm':
        # obs = np.delete(main_obs, np.s_[81:108])
        # obs = np.delete(obs, np.s_[108:135])
        obs = np.delete(main_obs, np.s_[27:54])

        obs = np.delete(obs, np.s_[30:33])

        obs = np.delete(obs, np.s_[45:54])
        obs = np.delete(obs, np.s_[39:42])

        #obs = np.delete(obs, np.s_[63:90])


    if leg_name is 'lr':
        # obs = np.delete(main_obs, np.s_[81:135])
        obs = np.delete(main_obs, np.s_[0:9])
        obs = np.delete(obs, np.s_[18:36])

        obs = np.delete(obs, np.s_[27])
        obs = np.delete(obs, np.s_[29:31])

        obs = np.delete(obs, np.s_[36:39])
        obs = np.delete(obs, np.s_[42:48])
        obs = np.delete(obs, np.s_[39:42])

        #obs = np.delete(obs, np.s_[63:81])
        #obs = np.delete(obs, np.s_[36:45])



    if leg_name is 'rf':
        # obs = np.delete(main_obs, np.s_[27:81])
        obs = np.delete(main_obs, np.s_[9:27])
        obs = np.delete(obs, np.s_[27:36])

        obs = np.delete(obs, np.s_[28:30])
        obs = np.delete(obs, np.s_[30])

        obs = np.delete(obs, np.s_[39:45])
        obs = np.delete(obs, np.s_[45:48])
        obs = np.delete(obs, np.s_[39:42])

        # obs = np.delete(obs, np.s_[81:90])
        # obs = np.delete(obs, np.s_[45:63])


    if leg_name is 'rm':
        # obs = np.delete(main_obs, np.s_[0:27])
        # obs = np.delete(obs, np.s_[27:54])
        obs = np.delete(main_obs, np.s_[0:27])
 
        obs = np.delete(obs, np.s_[27:30])
 
        obs = np.delete(obs, np.s_[36:45])
        obs = np.delete(obs, np.s_[39:42])

        #obs = np.delete(obs, np.s_[36:63])


    if leg_name is 'rr':
        # obs = np.delete(main_obs, np.s_[0:54])
        obs = np.delete(main_obs, np.s_[0:18])
        obs = np.delete(obs, np.s_[9:18])

        obs = np.delete(obs, np.s_[27:29])
        obs = np.delete(obs, np.s_[28])

        obs = np.delete(obs, np.s_[36:42])
        obs = np.delete(obs, np.s_[39:42])
        obs = np.delete(obs, np.s_[42:45])

        #obs = np.delete(obs, np.s_[63:72])
        #obs = np.delete(obs, np.s_[36:54])

    return obs

# Get dictionary from baselines/ppo2/defaults
defaults = get_learn_function_defaults('ppo2', 'phantomx_mlp')

env = gym.make('PhantomX-v0')

set_global_seeds(defaults['seed'])

alg_kwargs ={ 'num_layers': defaults['num_layers'], 'num_hidden': defaults['num_hidden'] }

nenvs = 1


nbatch = nenvs * defaults['nsteps']
nbatch_train = nbatch // defaults['nminibatches']


legs = ['lf', 'lm', 'lr', 'rf', 'rm', 'rr']
models = {}

def runner(leg, env):
    leg_env = gym.make('PhantomXLeg-v0')
    leg_env.set_info(env.info)
    leg_env.leg_name = leg
    policy = build_policy(leg_env, defaults['network'], **alg_kwargs)

    model = ppo2.Model(policy=policy, ob_space=leg_env.observation_space, ac_space=leg_env.action_space, nbatch_act=nenvs,
                    nbatch_train=nbatch_train,
                    nsteps=defaults['nsteps'], ent_coef=defaults['ent_coef'], vf_coef=defaults['vf_coef'],
                    max_grad_norm=defaults['max_grad_norm'])
    model.load('/tmp/training_data/dockerv1.3/PhantomX-v0/dppo2_mlp/2019-12-03_17h05min/' + leg + '/checkpoints/07030')
    obs = leg_env.reset()
    while True:
        action, value_estimate, next_state, neglogp = model.step(obs)
        obs, reward, done, _ = leg_env.step(action[0])
        time.sleep(1/1000)


for leg in legs:
    models[leg] = threading.Thread(target=runner, args=(leg, env))
    models[leg].start()
    #models[leg].load('/tmp/ros2learn/PhantomX-v0/dppo2_mlp/2019-12-10_14h25min/' + leg + '/checkpoints/00001')
    #model.save_model('/tmp/training_data/dockerv1.3/PhantomX-v0/dppo2_mlp/2019-12-03_17h05min/' + leg + '/checkpoints/model.ckp')
    #models[leg] = make_model()
    #models[leg].load_model('/tmp/training_data/dockerv1.3/PhantomX-v0/dppo2_mlp/2019-12-03_17h05min/' + leg + '/checkpoints/model.ckp')

loop = True
while loop:
    env.info.execute_action()
    env.info.execute_reset()
    time.sleep(1/1000)
