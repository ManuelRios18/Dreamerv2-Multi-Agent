import agent
import common
import pathlib


class Mob:

    def __init__(self, config, logdir, n_agents, prefix):
        self.config = config
        self.logdir = logdir
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
        self.weights_directory = pathlib.Path(logdir / "weights").expanduser()
        self.weights_directory.mkdir(parents=True, exist_ok=True)

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
            state, metrics = self.agents[player_id].train(next(dataset), state)
            new_states[player_id] = state
            agent_metrics[player_id] = metrics
        agent_metrics = {f"{player_id}_{metric_name}": metric_value
                         for player_id, value in agent_metrics.items() for metric_name, metric_value in value.items()}
        return new_states, agent_metrics

    def mob_policy(self, observations, states, mode):
        mob_actions = {}
        mob_states = {}
        for player_id, obs in observations.items():
            state = states[player_id]
            action, state = self.agents[player_id].policy(obs=obs, state=state, mode=mode)
            mob_actions[player_id] = action
            mob_states[player_id] = state
        return mob_actions, mob_states

    def load_agents(self):
        load_success = True
        for player_id in range(self.n_agents):
            path = self.weights_directory / f"agent_{player_id}_variables.pkl"
            if path.exists():
                self.agents[player_id].load(path)
            else:
                load_success = False

        return load_success

    def save_agents(self, prefix=""):
        for player_id in range(self.n_agents):
            self.agents[player_id].save(self.weights_directory / f"{prefix}agent_{player_id}_variables.pkl")

    def report(self, dataset):
        result = {}
        for player_id, data in dataset.items():
            report = self.agents[player_id].report(next(data))
            for k, v in report.items():
                result[f"{player_id}_{k}"] = v
        return result

