class ModelConfig(object):
    def __init__(
            self,
            vocab_size=30522,
            token_type_size=2,
            max_seq_len=4096,
            embed_size=128,
            hidden_size=512,
            num_heads=4,
            intermediate_size=2048,
            num_hidden_layers=6,
            num_hidden_groups=1,
            inner_group_num=1,
            initializer_range=0.02,
            layer_norm_epsilon=0.01,
            activation="gelu_new",
            dropout_rate=0.2,
            use_fft=True
    ):

        self.vocab_size = vocab_size
        self.token_type_size = token_type_size
        self.max_seq_len = max_seq_len
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.inner_group_num = inner_group_num
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_fft = use_fft
