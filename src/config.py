class Config:
    def __init__(self, **kwargs):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_name = "CartPole-v1"

        self.num_episodes = 50
        self.num_epochs_policy_network = 100
        self.render_every_n_epochs = 500
        self.log_params_every_n_epochs = 3

        self.policy_hidden_size = 4
        self.policy_learning_rate = 0.01

        self.value_hidden_size = 16
        self.value_learning_rate = 0.1
        self.num_epochs_value_network = 1

        self.weight_kind = 'gae' # 'r' for 'returns', 'fr' for 'future returns', 'dfr' for 'discounted future returns',
        # 'gae' for 'generalized advantage estimates', 'td' for 'temporal difference errors', 'dfrb' for 'discounted future returns - baseline'
        self.avg_kind = 'a' # 'a' for 'all' and 't' for 'trajectories' 
        self.gamma_gae = 0.99
        self.lambda_gae = 0.96

        self.log_dir = "runs\\test"
        self.base_log_dir = "runs"
        self.experiment_group_name = "exp_vf_lr"

        self.run_tag = ""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
            self.run_tag += key + "_" + str(value) + "_"