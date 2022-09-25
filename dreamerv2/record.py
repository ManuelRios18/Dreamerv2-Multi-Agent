import collections
import logging
import os
import pathlib
import re
import sys
import warnings


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import mob
import common

from common import Config
from common import GymWrapper
from common import RenderImage
from common import TerminalOutput
from common import JSONLOutput
from common import TensorBoardOutput

configs = yaml.safe_load(
    (pathlib.Path(__file__).parent / 'configs.yaml').read_text())
defaults = common.Config(configs.pop('defaults'))


def record(env, testing_env, config, n_episodes):

  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  outputs = [common.TerminalOutput()]

  if config.precision == 16:
    import tensorflow.keras.mixed_precision as prec
    prec.set_global_policy(prec.Policy('mixed_float16'))
    print("Setting mixed_float16")

  n_agents = env._num_players
  agents_prefix = "player_"
  agents_mob = mob.Mob(config, logdir, n_agents, agents_prefix, load_train_ds=False)
  agents_mob.initialize_recording()

  step = common.Counter(agents_mob.record_replays[agents_prefix+str(0)].stats['total_steps'])
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)

  def per_episode(ep, mode):
    length = len(ep["player_0"]['reward']) - 1
    score = ep["metrics"]["efficiency"]
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    for metric_name, metric_value in ep["metrics"].items():
      logger.scalar(f'{mode}_{metric_name}', metric_value)
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'{mode}_sum_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'{mode}_mean_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'{mode}_max_{key}', ep[key].max(0).mean())
    replay = agents_mob.record_replays[agents_prefix+str(0)]
    logger.add(replay.stats, prefix=mode)
    logger.write()

  def wrap_env(base_env):
    wrapped_env = common.GymWrapperMultiAgent(base_env)
    wrapped_env = common.ResizeImageMultiAgent(wrapped_env)
    if hasattr(wrapped_env.act_space['action'], 'n'):
      train_env = common.OneHotActionMultiAgent(wrapped_env)
    else:
      train_env = common.NormalizeAction(wrapped_env)
    wrapped_env = common.TimeLimit(train_env, config.time_limit)
    return wrapped_env

  eval_env = wrap_env(testing_env)

  eval_driver = common.MultiAgentDriver([eval_env], n_agents, agents_prefix)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  eval_driver.on_episode(agents_mob.add_episode_record)

  print('Loading agents.')
  agents_mob.create_agents(eval_env.obs_space, eval_env.act_space, step)


  eval_datasets = agents_mob.get_datasets(mode="eval")

  train_agents = common.CarryOverStateMultiAgent(agents_mob.train_mob, n_agents, agents_prefix)
  train_agents(eval_datasets)
  agents_mob.load_agents()
  eval_policy = lambda *args: agents_mob.mob_policy(*args, mode='eval')

  while step < n_episodes:
    logger.write()
    eval_driver(eval_policy, episodes=config.eval_eps)
