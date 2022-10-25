# from __future__ import division
# import torch
# from transformers import RoFormerLayer, TFDistilBertModel, TFBertModel, TFT5Model, TFTransfoXLModel, \
#     TFAlbertModel, T5Config, TFMobileBertModel, MobileBertConfig, FNetModel, FNetConfig, IBertModel, IBertConfig,\
#     TFBertForMaskedLM, BertConfig, BertTokenizerFast, BertTokenizer
import tensorflow as tf
# import numpy as np
# import torch
# from scipy import linalg
import os


class CustomModel(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        outputs = tf.keras.layers.Dense(128)(inputs)
        outputs = tf.keras.layers