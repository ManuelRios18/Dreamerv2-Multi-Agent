import agent
import common


class Mob:

    def __init__(self, config, logdir, n_agents, prefix):
        self.config = config
        self.n_agents = n_agents
        self.prefix = prefix
        self.train_replays = {prefix + str(player_id): common.Replay(logdir / f"{prefix}{player_id}_train_episodes",
                                                                     **config.replay)
                              for player_id in n_agents}

        self.eval_replays = {prefix + str(player_id):
                             common.Replay(logdir / f"{prefix}{player_id}_eval_episodes",
                                           **dict(
                                               capacity=config.replay.capacity // 10,
                                               minlen=config.dataset.length,
                                               maxlen=config.dataset.length)
                                           )
                             for player_id in n_agents}
        self.agents = None

    def add_steps_train(self, transitions, worker=0):
        """

        :param transitions: {"player_0": tran, "player_1": tran}
        :param worker:
        :return:
        """
        for player_id, transition in transitions.items():
            self.train_replays[player_id].add_step(transition, worker)

    def add_episode_eval(self, episode):
        for player_id, ep in episode.items():
            self.eval_replays[player_id].add_episode(ep)

    def create_agents(self, obs_space, act_space, step):
        self.agents = {}
        for player_num in range(self.n_agents):
            self.agents[f"{self.prefix}{player_num}"] = agent.Agent(self.config, obs_space, act_space, step)

    def get_datasets(self, mode="train"):
        replays = dict(train=self.train_replays, eval=self.eval_replays)[mode]
        datasets = {player_id: iter(replays[player_id].dataset(**self.config.dataset))
                    for player_id in replays.keys()}
        return datasets

    def train_mob(self, training_data, agents_states):
        new_states, agent_metrics = {}, {}
        for player_id, data in training_data.items():
            dataset = next(data)
            state = agents_states[player_id]
            state, metrics = self.agents[player_id].train(next(dataset))
            new_states[player_id] = state
            agent_metrics[player_id] = metrics
        return new_states, metrics


