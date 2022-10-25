import tensorflow as tf


def get_initializer(initializer_range):
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def get_activation(activation_string):
    # Clip the range of possible GeLU outputs between [-10, 10]. For more information on this trick,
    # please refer to https://arxiv.org/abs/2004.09602
    def gelu_10(x):
        return tf.clip_by_value(x, -10, 10)

    # The smoother version of the GELU. For more information,
    # please refer to the original paper https://arxiv.org/abs/1606.0841.
    def gelu_new(x):
        return tf.nn.gelu(x, approximate=True)

    def gelu_fast(x):
        x = tf.convert_to_tensor(x)
        coeff1 = tf.cast(0.044715, x.dtype)
        coeff2 = tf.cast(0.7978845608, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(x * coeff2 * (1.0 + coeff1 * x * x)))

    def quick_gelu(x):
        x = tf.convert_to_tensor(x)
        coeff = tf.cast(1.702, x.dtype)
        return x * tf.math.sigmoid(coeff * x)

    # Gated Linear Unit. Split the input x into two halves a and b, and return a * sigmoid(b).
    # For more detail, please refer to https://arxiv.org/abs/1612.08083.
    def glu(x):
        x = tf.convert_to_tensor(x)
        a, b = tf.split(x, 2, axis=-1)
        return a * tf.nn.sigmoid(b)

    string2func = {
        "tanh": tf.nn.tanh,
        "relu": tf.nn.relu,
        "relu6": tf.nn.relu6,
        "leaky_relu": tf.nn.leaky_relu,
        "gelu": tf.nn.gelu,
        "gelu_10": gelu_10,
        "gelu_new": gelu_new,
        "gelu_fast": gelu_fast,
        "quick_gelu": quick_gelu,
        "glu": glu,
        "elu": tf.nn.elu,
        "selu": tf.nn.selu,
        "softsign": tf.nn.softsign,
        "softplus": tf.nn.softplus,
        "silu": tf.nn.silu,  # A special case of swish which beta is equal to 1.
        "swish": tf.nn.swish
    }

    if string2func[activation_string] is not None:
        return string2func[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in {list(string2func.keys())}")


def dummy_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)
