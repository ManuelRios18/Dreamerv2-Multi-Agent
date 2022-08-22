import numpy as np


class Driver:

  def __init__(self, envs, **kwargs):
    self._envs = envs
    self._kwargs = kwargs
    self._on_steps = []
    self._on_resets = []
    self._on_episodes = []
    self._act_spaces = [env.act_space for env in envs]
    self.reset()
    self.episode_rewards = []

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_reset(self, callback):
    self._on_resets.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def reset(self):
    self._obs = [None] * len(self._envs)
    self._eps = [None] * len(self._envs)
    self._state = None
    self.episode_rewards = []

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    self.episode_rewards = []
    while step < steps or episode < episodes:
      # This line fill obs if is the first time or ans episode ended
      obs = {
          i: self._envs[i].reset()
          for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
      # This block fills the first transitions with the corresponding observation and a action vector of zeros.
      # It is executed only if obs != {} i.e first time or episoded ended and has restarted.
      for i, ob in obs.items():
        self._obs[i] = ob() if callable(ob) else ob
        act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]
        self._eps[i] = [tran]
      # This line concatenates the observation from various envs
      obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
      # Use the policy to get the agent actions, note that policy must receive the last states.
      actions, self._state = policy(obs, self._state, **self._kwargs)
      # Transform actions into a list of numpy arrays: [{'action': array([0., 0., 0., 0., 0., 0., 1., 0.], dtype=float32)}]
      actions = [
          {k: np.array(actions[k][i]) for k in actions}
          for i in range(len(self._envs))]
      assert len(actions) == len(self._envs)
      # This line executes the actions on the environments and transform obs into a list of the osbervatins,
      # one list item for env
      obs = [e.step(a) for e, a in zip(self._envs, actions)]
      obs = [ob() if callable(ob) else ob for ob in obs]
      for i, (act, ob) in enumerate(zip(actions, obs)):
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
        self._eps[i].append(tran)
        step += 1
        if ob['is_last']:
          ep = self._eps[i]
          ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
          ep_reward = float(ep['reward'].astype(np.float64).sum())
          self.episode_rewards.append(ep_reward)
          [fn(ep, **self._kwargs) for fn in self._on_episodes]
          episode += 1
      self._obs = obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
      return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
      return value.astype(np.uint8)
    return value
