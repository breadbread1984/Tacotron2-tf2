#!/usr/bin/python3

import tensorflow as tf;

def Tacotron2(enc_filters = 512, kernel_size = 5, enc_layers = 3, drop_rate = 0.5, enc_lstm_units = 256):

  inputs = tf.keras.Input((None,)); # inputs.shape = (batch, seq_length)
  results = inputs;
  # 1) tacotron encoder cell
  # 1.1) convolutional layers
  # output shape = (batch, seq_length, enc_filters)
  for i in range(enc_layers):
    results = tf.keras.layers.Conv1D(filters = enc_filters, kernel_size = kernel_size, padding = 'same', activation = tf.keras.layers.ReLU())(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.Dropout(rate = drop_rate)(results);
  # 1.2) rnn layers (will use zoneout LSTM instead when it is available in tf.keras)
  # output shape = (batch, seq_length, 2 * enc_lstm_units)
  results = tf.keras.layers.Bidirectional(
    layer = tf.keras.layers.LSTM(enc_lstm_units, return_sequences = True), 
    backward_layer = tf.keras.LSTM(enc_lstm_units, return_sequences = True, go_backwards = True),
    merge_mode = 'concat')(results);
  # 2) tacotron decoder cell
  # 2.1) attention decoder prenet
  
