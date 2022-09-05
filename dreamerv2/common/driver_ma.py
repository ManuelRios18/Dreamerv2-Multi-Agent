import numpy as np


class MultiAgentDriver:

  def __init__(self, envs, n_agents, agents_prefix, **kwargs):
    self._envs = envs
    self.n_agents = n_agents
    self.agents_prefix = agents_prefix
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
    self._state = {f"{self.agents_prefix}{player_id}": None for player_id in range(self.n_agents)}
    self.episode_rewards = []

  def get_episode_metrics(self, current_episode, T):
    n_zaps = current_episode._env._env._env.world_data["n_zaps"]
    consumption_by_player = current_episode._env._env._env.world_data["consumption"]

    total_consumption = np.sum(consumption_by_player)
    efficiency = total_consumption/self.n_agents

    peacefulness = (self.n_agents * T - n_zaps) / self.n_agents

    mad = np.abs(np.subtract.outer(consumption_by_player, consumption_by_player)).mean()
    rmad = mad / np.mean(consumption_by_player)
    gini_index = 0.5 * rmad
    equality = 1-gini_index

    return {"efficiency": efficiency, "peacefulness": peacefulness, "equality": equality}

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    self.episode_rewards = []
    while step < steps or episode < episodes:
      obs = {
          i: self._envs[i].reset()
          for i, ob in enumerate(self._obs) if ob is None or ob[f"{self.agents_prefix}0"]['is_last']}

      for i, observations in obs.items():
        transitions = {}
        self._obs[i] = observations() if callable(observations) else observations
        for player_id, ob in observations.items():
          act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
          tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
          transitions[player_id] = tran
        [fn(transitions, worker=i, **self._kwargs) for fn in self._on_resets]
        self._eps[i] = [transitions]

      obs_keys = self._obs[0][f"{self.agents_prefix}0"].keys()
      players_ids = self._obs[0].keys()
      obs = {}
      for player_id in players_ids:
        obs[player_id] = {k: np.stack([obs[player_id][k] for obs in self._obs]) for k in obs_keys}

      actions, self._state = policy(obs, self._state, **self._kwargs)
      actions = [{player_id:
                    {k: np.array(actions[player_id][k][i]) for k in actions[player_id]}
                  for player_id in actions.keys()} for i in range(len(self._envs))]
      assert len(actions) == len(self._envs)
      obs = [e.step(a) for e, a in zip(self._envs, actions)]
      obs = [ob() if callable(ob) else ob for ob in obs]
      for i, (mob_act, mob_ob) in enumerate(zip(actions, obs)):
        transitions = {}
        for player_id, player_ob in mob_ob.items():
          act = mob_act[player_id]
          tran = {k: self._convert(v) for k, v in {**player_ob, **act}.items()}
          transitions[player_id] = tran
        [fn(transitions, worker=i, **self._kwargs) for fn in self._on_steps]
        self._eps[i].append(transitions)
        step += 1
        if mob_ob[f"{self.agents_prefix}0"]['is_last']:
          ep = self._eps[i]
          ep_len = len(ep) - 1
          ep = {player_id: {k: self._convert([t[player_id][k] for t in ep]) for k in ep[0]["player_0"]} for player_id in ep[0].keys()}
          ep_metrics = self.get_episode_metrics(self._envs[i], ep_len)
          ep_reward = ep_metrics["efficiency"]
          self.episode_rewards.append(ep_reward)
          ep["metrics"] = ep_metrics
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
