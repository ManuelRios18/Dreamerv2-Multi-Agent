import common


class Mob:

    def __init__(self, config, logdir, n_agents, prefix):

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
