import math

import tensorflow as tf

from scipy import linalg
from functools import partial

from data_adapter import unpack_data
from model_utils import (
    get_initializer,
    get_activation,
    dummy_loss
)


def get_sinusoidal_position_embedding(seq_len, hidden_size):
    pos_seq = tf.range(0, seq_len)
    pos_seq = tf.cast(pos_seq, dtype=tf.float32)

    inp_freq = 1 / 10000 ** (tf.range(0, hidden_size, 2) / hidden_size)
    inp_freq = tf.cast(inp_freq, dtype=pos_seq.dtype)

    pos_emb = tf.einsum("i,j->ij", pos_seq, inp_freq)
    pos_emb = tf.concat([tf.sin(pos_emb), tf.cos(pos_emb)], axis=-1)

    return pos_emb  # [seq_len, hidden_size]


def _two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    seq_length = x.shape[1]
    matrix_dim_one = matrix_dim_one[:seq_length, :seq_length]
    x = tf.cast(x, tf.complex64)
    return tf.einsum("bij,jk,ni->bnk", x, matrix_dim_two, matrix_dim_one)


def two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    return _two_dim_matmul(x, matrix_dim_one, matrix_dim_two)


class FasterBERTEmbedding(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.initializer_range = config.initializer_range

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(rate=config.dropout_rate, name="dropout")

    def build(self, input_shape):
        self.word_weights = self.add_weight(
            name="word_weights",
            shape=(self.config.vocab_size, self.config.embed_size),
            initializer=get_initializer(initializer_range=self.initializer_range)
        )
        self.type_weights = self.add_weight(
            name="word_type_weights",
            shape=(self.config.token_type_size, self.config.embed_size),
            initializer=get_initializer(initializer_range=self.initializer_range)
        )
        super().build(input_shape)

    def call(self, input_ids=None, token_type_ids=None, input_embeds=None, training=False):
        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(input_embeds)[:-1]

        if input_embeds is None:
            input_embeds = tf.gather(self.word_weights, tf.cast(input_ids, dtype=tf.int32))

        if token_type_ids is None:
            token_type_ids = tf.zeros(input_shape, tf.int32)
        token_type_embeds = tf.gather(self.type_weights, tf.cast(token_type_ids, dtype=tf.int32))
        embeds = input_embeds + token_type_embeds
        embeds = self.dropout(embeds, training=training)
        embeds = self.layer_norm(embeds)
        return embeds


class FasterBERTIntermediate(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.intermediate = tf.keras.layers.Dense(
            config.intermediate_size,
            name="intermediate",
            kernel_initializer=get_initializer(config.initializer_range)
        )
        self.activation = get_activation("gelu")

    def call(self, hidden_states):
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class FasterBERTOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            name="dense",
            kernel_initializer=get_initializer(config.initializer_range)
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon)

    def call(self, hidden_states, input_tensor=None, training=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states


class FasterBERTFeedForward(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.intermediate = FasterBERTIntermediate(config, name="intermediate")
        self.intermediate_output = FasterBERTOutput(config, name="intermediate_output")

    def call(self, hidden_states, training=False):
        intermediate_outputs = self.intermediate(hidden_states)
        final_outputs = self.intermediate_output(intermediate_outputs, hidden_states, training=training)
        return final_outputs


class FasterBERTBasicFourierTransform(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.basic_fourier_transform = self._init_basic_fourier_transform()

    def _init_basic_fourier_transform(self):
        # On GPUs/CPUs: The native FFT implementation is optimal for all sequence lengths.
        # On TPUs: For relatively shorter sequences, it is faster to pre-compute the
        # DFT matrix and then compute Fourier Transform using matrix multiplications.
        # For longer sequences, the FFT is faster, provided the MAX_SEQ_LENGTH is a
        # power of 2.
        if self.config.use_fft:
            if self.config.max_seq_len <= 4096 or math.log2(self.config.max_seq_len).is_integer():
                return tf.signal.fft2d
            else:
                raise ValueError("For large input sequence lengths (>4096), the maximum input "
                                 "sequence length must be a power of 2 to take advantage of FFT "
                                 "optimizations. ")
        else:
            dft_mat_hidden = tf.cast(linalg.dft(self.config.hidden_size), tf.complex64)
            dft_mat_seq = tf.cast(linalg.dft(self.config.max_seq_len), tf.complex64)
            return partial(
                two_dim_matmul, matrix_dim_one=dft_mat_seq, matrix_dim_two=dft_mat_hidden
            )

    def call(self, hidden_states):
        hidden_states = tf.cast(hidden_states, tf.complex64)
        outputs = self.basic_fourier_transform(hidden_states)
        # take the real part of the complex
        outputs = tf.math.real(outputs)
        return outputs


class FasterBERTFourierOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon)

    def call(self, hidden_states, input_tensor=None):
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class FasterBERTFourierTransform(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.fourier_transform = FasterBERTBasicFourierTransform(config, name="fourier_transform")
        self.fourier_output = FasterBERTFourierOutput(config, name="fourier_output")

    def call(self, hidden_states, training=False):
        fourier_outputs = self.fourier_transform(hidden_states)
        fourier_outputs = self.fourier_output(fourier_outputs, hidden_states, training=training)
        return fourier_outputs


class FasterBERTSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        assert config.hidden_size % config.num_heads == 0

        self.num_heads = config.num_heads
        self.head_size = int(config.hidden_size / self.num_heads)

        # Although hidden_size is equal to all_head_size, we usually use all_head_size instead of
        # hidden_size in self attention. Finally, convert it into hidden_size.
        self.all_head_size = self.num_heads * self.head_size

        self.query = tf.keras.layers.Dense(
            self.all_head_size,
            name="query",
            kernel_initializer=get_initializer(config.initializer_range)
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size,
            name="key",
            kernel_initializer=get_initializer(config.initializer_range)
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size,
            name="value",
            kernel_initializer=get_initializer(config.initializer_range)
        )

        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            name="dense",
            kernel_initializer=get_initializer(config.initializer_range)
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.dropout_rate, name="dropout")

    def transpose_for_scores(self, tensor, batch_size):
        tensor = tf.reshape(tensor, (batch_size, -1, self.num_heads, self.head_size))
        tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
        return tensor

    def call(self, hidden_states, attention_mask=None, position_embeddings=None, output_attentions=False, training=False):
        batch_size, seq_len = tf.shape(hidden_states)[0], tf.shape(hidden_states)[1]

        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        # Split to multiple heads.
        query_layer = self.transpose_for_scores(query_layer, batch_size)
        key_layer = self.transpose_for_scores(key_layer, batch_size)
        value_layer = self.transpose_for_scores(value_layer, batch_size)

        # Apply rotary position embedding.
        if position_embeddings is not None:
            # sin [batch_size, num_heads, seq_len, hidden_size // 2]
            # cos [batch_size, num_heads, seq_len, hidden_size // 2]
            sin, cos = tf.split(position_embeddings, 2, -1)
            # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
            # sin_pos [batch_size, num_heads, seq_len, hidden_size // 2]
            sin_pos = tf.stack([sin, sin], axis=-1)
            # [batch_size, num_heads, seq_len, hidden_size // 2, 2] -> [batch_size, num_heads, seq_len, hidden_size]
            sin_pos = tf.reshape(sin_pos, position_embeddings.shape)
            # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
            # cos_pos [batch_size, num_heads, seq_len, hidden_size // 2]
            cos_pos = tf.stack([cos, cos], axis=-1)
            cos_pos = tf.reshape(cos_pos, position_embeddings.shape)

            # assume q,k,v is a token vector, and qi is an element is this token vector
            # q [q0,q1,q2,q3......qd-2,qd-1] -> [-q1,q0,-q3,q2......-qd-1,qd-2]
            # query_layer1 [batch_size, num_heads, seq_len, hidden_size // 2, 2]
            query_layer1 = tf.stack([-query_layer[:, :, :, 1::2], query_layer[:, :, :, ::2]], axis=-1)
            # query_layer1
            # [batch_size, num_heads, seq_len, hidden_size // 2, 2] -> [batch_size, num_heads, seq_len, hidden_size]
            query_layer1 = tf.reshape(query_layer1, query_layer.shape)
            # query layer with position information.
            query_layer = query_layer * cos_pos + query_layer1 * sin_pos

            # key_layer1
            key_layer1 = tf.stack([-key_layer[:, :, :, 1::2], key_layer[:, :, :, ::2]], axis=-1)
            key_layer1 = tf.reshape(key_layer1, key_layer.shape)
            key_layer = key_layer * cos_pos + key_layer1 * sin_pos

            if value_layer is not None:
                # value_layer1
                value_layer1 = tf.stack([-value_layer[:, :, :, 1::2], value_layer[:, :, :, ::2]], axis=-1)
                value_layer1 = tf.reshape(value_layer1, value_layer.shape)
                value_layer = value_layer * cos_pos + value_layer1 * sin_pos

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.head_size, attention_scores.dtype))

        if attention_mask is not None:
            attention_scores = tf.add(attention_scores, attention_mask * -1e8)

        attention_probs = tf.nn.softmax(attention_scores + 1e-8, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        attention_outputs = tf.matmul(attention_probs, value_layer)
        attention_outputs = tf.transpose(attention_outputs, perm=[0, 2, 1, 3])
        attention_outputs = tf.reshape(attention_outputs, (batch_size, seq_len, -1))

        outputs = (attention_outputs, attention_probs) if output_attentions else (attention_outputs,)
        return outputs


class FasterBERTAttentionOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            name="dense",
            kernel_initializer=get_initializer(config.initializer_range)
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.dropout_rate, name="dropout")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="layer_norm")

    def call(self, hidden_states, input_tensor=None, training=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states


class FasterBERTAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = FasterBERTSelfAttention(config, name="self_attention")
        self.attention_output = FasterBERTAttentionOutput(config, name="attention_output")

    def call(self, hidden_states, attention_mask=None, position_embeddings=None, output_attentions=None, training=False):
        attention_outputs = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            training=training
        )
        outputs = self.attention_output(
            attention_outputs[0],
            input_tensor=hidden_states,
            training=training
        )
        # Add attention if we need it.
        outputs = (outputs, )
        if output_attentions:
            outputs = outputs + attention_outputs[1:]
        return outputs


class FasterBERTLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.attention = FasterBERTAttention(config, name="attention")
        self.fourier_transform = FasterBERTFourierTransform(config, name="fourier_transform")
        self.feedforward = FasterBERTFeedForward(config, name="feedforward")

    def call(
            self,
            hidden_states,
            attention_mask=None,
            position_embeddings=None,
            output_attentions=None,
            training=False
    ):
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            training=training
        )
        # add attention weights if parameter output_attention is true.
        attention_weights = attention_outputs[1:]

        fourier_transform_outputs = self.fourier_transform(attention_outputs[0])
        feedforward_outputs = self.feedforward(fourier_transform_outputs)

        outputs = (feedforward_outputs,)
        if output_attentions:
            outputs = outputs + attention_weights
        return outputs


class FasterBERTLayerGroup(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.layers = [FasterBERTLayer(config, name=f"layer_{i}") for i in range(config.inner_group_num)]

    def call(
            self,
            hidden_states,
            attention_mask=None,
            position_embeddings=None,
            output_hidden_states=False,
            output_attentions=False,
            training=False
    ):
        layer_hidden_states = ()
        layer_attentions = ()
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states, )
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                training=training
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                layer_attentions = layer_attentions + (layer_outputs[1], )
        # Add the last layer.
        if output_hidden_states:
            layer_hidden_states = layer_hidden_states + (hidden_states, )
        # The hidden states of per inner layer is not use. We just take both
        # the last layer hidden states and per layer attentions.
        return tuple(v for v in [hidden_states, layer_hidden_states, layer_attentions] if v is not None)


class FasterBERTEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.num_heads = config.num_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups

        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense"
        )
        self.position_embeddings = get_sinusoidal_position_embedding(config.max_seq_len, config.hidden_size)
        self.layer_groups = [
            FasterBERTLayerGroup(config, name=f"group_{i}") for i in range(config.num_hidden_groups)
        ]

    def call(
            self,
            hidden_states,
            attention_mask=None,
            position_embeddings=None,
            output_hidden_states=False,
            output_attentions=False,
            training=False
    ):
        # hidden_states [batch_size, seq_len, embed_size] -> [batch_size, seq_len, hidden_size]
        # actually, embed_size << hidden_size.
        hidden_states = self.dense(hidden_states)

        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        position_embeddings = self.position_embeddings[None, :seq_len, :]
        position_embeddings = tf.tile(position_embeddings, (batch_size, 1, 1))
        # [batch_size, seq_len, hidden_size] -> [batch_size, num_heads, seq_len, head_size]
        position_embeddings = tf.reshape(position_embeddings, (batch_size, seq_len, self.num_heads, -1))
        position_embeddings = tf.transpose(position_embeddings, perm=[0, 2, 1, 3])

        all_hidden_states = ()
        all_attentions = ()

        for i in range(self.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            group_idx = int(i / (self.num_hidden_layers / self.num_hidden_groups))
            layer_group_outputs = self.layer_groups[group_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                training=training
            )
            hidden_states = layer_group_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + layer_group_outputs[-1]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


class FasterBERTMainLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.embedding = FasterBERTEmbedding(config, name="embedding")
        self.encoder = FasterBERTEncoder(config, name="encoder")

    def call(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            input_embeds=None,
            output_hidden_states=None,
            output_attentions=None,
            training=False,
    ):
        if input_ids is not None and input_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = tf.shape(input_ids)
        elif input_embeds is not None:
            input_shape = tf.shape(input_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or input_embeds")

        batch_size, seq_len = input_shape

        if attention_mask is None:
            attention_mask = tf.ones((batch_size, seq_len), tf.float32)
        else:
            attention_mask = tf.cast(attention_mask, tf.float32)

        # make the mask broadcast to [batch_size, num_heads, mask_seq_len, mask_seq_len].
        extended_attention_mask = tf.reshape(
            attention_mask, (attention_mask.shape[0], 1, 1, attention_mask.shape[1])
        )

        embedding_outputs = self.embedding(input_ids, token_type_ids, input_embeds, training)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions.
        one_cst = tf.constant(1.0, dtype=embedding_outputs.dtype)
        extend_attention_mask = tf.subtract(one_cst, extended_attention_mask, embedding_outputs.dtype)

        outputs = self.encoder(
            embedding_outputs,
            attention_mask=extend_attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            training=training
        )
        # outputs (last_hidden_states, all_hidden_states, attentions)
        return outputs


class FasterBERTModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.faster_bert = FasterBERTMainLayer(config, name="faster_bert")

    def call(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            input_embeds=None,
            output_hidden_states=False,
            output_attentions=False,
            training=False
    ):
        outputs = self.faster_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            input_embeds=input_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            training=training
        )
        return outputs


class FasterBERTMLMHead(tf.keras.layers.Layer):
    def __init__(self, config, embeddings, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.embed_size = config.embed_size

        self.dense = tf.keras.layers.Dense(
            config.embed_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense"
        )
        self.activation = get_activation("gelu")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="layer_norm")

        self.embeddings = embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="bias",
            shape=(self.vocab_size, ),
            initializer="zeros"
        )
        super().build(input_shape)

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        seq_len = hidden_states.shape[1]
        hidden_states = tf.reshape(hidden_states, (-1, self.embed_size))
        hidden_states = tf.matmul(hidden_states, self.embeddings.weight, transpose_b=True)
        hidden_states = tf.reshape(hidden_states, (-1, seq_len, self.vocab_size))
        hidden_states = tf.nn.bias_add(hidden_states, bias=self.bias)
        return hidden_states


class FasterBERTForMaskedLM(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.faster_bert = FasterBERTMainLayer(config, name="faster_bert")
        self.predictions = FasterBERTMLMHead(config, self.faster_bert.embedding, name="predictions")

    @staticmethod
    def compute_mlm_loss(labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )
        unmasked_loss = loss_fn(y_true=tf.nn.relu(labels), y_pred=logits)
        loss_mask = tf.cast(labels != -100, dtype=unmasked_loss.dtype)
        masked_loss = unmasked_loss * loss_mask
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return reduced_masked_loss

    def call(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            input_embeds=None,
            output_hidden_states=False,
            output_attentions=False,
            labels=None,
            training=False
    ):
        outputs = self.faster_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            input_embeds=input_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            training=training
        )
        sequence_outputs = outputs[0]
        prediction_scores = self.predictions(sequence_outputs)
        loss = None if labels is None else self.compute_mlm_loss(labels=labels, logits=prediction_scores)
        outputs = (prediction_scores, ) + outputs[1:]

        if loss is not None:
            return (loss,) + outputs  # (loss, prediction_scores, all_hidden_states, all_attentions)
        else:
            return outputs  # (prediction_scores, all_hidden_states, all_attentions)

    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        jit_compile=None,
        **kwargs,
    ):
        loss = dummy_loss
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            **kwargs
        )

    def train_step(self, data):
        input_names = ["input_ids", "attention_mask", "token_type_ids",
                       "input_embeds", "output_attentions", "output_hidden_states",
                       "labels", "training"]
        x, y, sample_weight = unpack_data(data)
        inputs = {key: val for key, val in x.items() if key in input_names}

        with tf.GradientTape() as tape:
            y_pred = self(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                input_embeds=inputs["input_embeds"],
                labels=inputs["labels"],
                training=True
            )

            # y_pred[0] loss
            loss = self.compiled_loss(
                y_true=y_pred[0],
                y_pred=y_pred[0],
                sample_weight=sample_weight,
                regularization_losses=self.losses
            )

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred[1], sample_weight)
        return {m.name: m.result() for m in self.metrics}