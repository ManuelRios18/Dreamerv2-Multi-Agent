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


def train(env, testing_env, config, outputs=None):

  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  outputs = outputs or [
      common.TerminalOutput(),
      common.JSONLOutput(config.logdir),
      common.TensorBoardOutput(config.logdir),
  ]

  if config.precision == 16:
    import tensorflow.keras.mixed_precision as prec
    prec.set_global_policy(prec.Policy('mixed_float16'))
    print("Setting mixed_float16")

  train_replay = common.Replay(logdir / 'train_episodes', **config.replay)
  eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
    capacity=config.replay.capacity // 10,
    minlen=config.dataset.length,
    maxlen=config.dataset.length))
  step = common.Counter(train_replay.stats['total_steps'])
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  should_train = common.Every(config.train_every)
  should_log = common.Every(config.log_every)
  should_video_train = common.Every(config.log_every)
  should_video_eval = common.Every(config.eval_every)
  should_expl = common.Until(config.expl_until)

  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'{mode}_sum_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'{mode}_mean_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'{mode}_max_{key}', ep[key].max(0).mean())
    if should_video_train(step):
      for key in config.log_keys_video:
        logger.video(f'{mode}_policy_{key}', ep[key])
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(replay.stats, prefix=mode)
    logger.write()

  def wrap_env(base_env):
    wrapped_env = common.GymWrapper(base_env)
    wrapped_env = common.ResizeImage(wrapped_env)
    if hasattr(wrapped_env.act_space['action'], 'n'):
      train_env = common.OneHotAction(wrapped_env)
    else:
      train_env = common.NormalizeAction(wrapped_env)
    wrapped_env = common.TimeLimit(train_env, config.time_limit)
    return wrapped_env

  train_env = wrap_env(env)
  eval_env = wrap_env(testing_env)

  train_driver = common.Driver([train_env])
  train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
  train_driver.on_step(lambda tran, worker: step.increment())
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)

  eval_driver = common.Driver([eval_env])
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  eval_driver.on_episode(eval_replay.add_episode)

  prefill = max(0, config.prefill - train_replay.stats['total_steps'])
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    random_agent = common.RandomAgent(train_env.act_space)
    train_driver(random_agent, steps=prefill, episodes=1)
    eval_driver(random_agent, episodes=1)
    train_driver.reset()
    eval_driver.reset()

  print('Create agent.')
  agnt = agent.Agent(config, train_env.obs_space, train_env.act_space, step)
  train_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(train_dataset))
  if (logdir / 'variables.pkl').exists():
    agnt.load(logdir / 'variables.pkl')
  else:
    print('Pretrain agent.')
    for _ in range(config.pretrain):
      train_agent(next(train_dataset))
  train_policy = lambda *args: agnt.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  eval_policy = lambda *args: agnt.policy(*args, mode='eval')

  def train_step(tran, worker):
    if should_train(step):
      for _ in range(config.train_steps):
        mets = train_agent(next(train_dataset))
        [metrics[key].append(value) for key, value in mets.items()]
    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.add(agnt.report(next(train_dataset)))
      logger.write(fps=True)
  train_driver.on_step(train_step)
  best_agent_score = -float("inf")
  while step < config.steps:
    logger.write()
    print('Start evaluation.')
    logger.add(agnt.report(next(eval_dataset)), prefix='eval')
    eval_driver(eval_policy, episodes=config.eval_eps)
    agent_score = np.mean(eval_driver.episode_rewards)
    if agent_score > best_agent_score:
      best_agent_score = agent_score
      agent_id = f"ba_{step.value}"
      agnt.save(logdir / f'{agent_id}_variables.pkl')
      print(f"Saving best agent with score {agent_score}")
    print('Start training.')
    train_driver(train_policy, steps=config.eval_every)
    agnt.save(logdir / 'variables.pkl')
