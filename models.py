#!/usr/bin/python3

import tensorflow as tf;

class ZoneoutLSTMCell(tf.keras.layers.Layer):

  def __init__(self, units, zoneout_h = 0., zoneout_c = 0., **kwargs):

    self.units = units;
    self.zoneout_h = zoneout_h;
    self.zoneout_c = zoneout_c;
    self.state_size = [tf.TensorShape([self.units,]), tf.TensorShape([self.units,])];
    self.output_size = tf.TensorShape([self.units,]);
    super(ZoneoutLSTMCell, self).__init__(**kwargs);

  def zoneout(self, v, prev_v, drop_rate = 0.):

    diff = v - prev_v; # diff.shape = (batch, hidden_dim)
    diff = tf.keras.backend.in_train_phase(tf.keras.backend.dropout(diff, drop_rate), diff);
    return prev_v + diff * (1 - drop_rate);

  def build(self, input_shape):

    self.W = self.add_weight(shape = (input_shape[-1], 4 * self.units)); # shape = (input_dim, 4 * hidden_dim)
    self.U = self.add_weight(shape = (self.units, 4 * self.units)); # shape = (hidden_dim, 4 * hidden_dim)
    self.b = self.add_weight(shape = (4 * self.units,)); # shape = (4 * hidden_dim)

  def call(self, inputs, states):

    hidden_tm1 = states[0]; # hidden_tm1.shape = (batch, hidden_dim)
    cell_tm1 = states[1]; # cell_tm1.shape = (batch, hidden_dim)
    z = tf.linalg.matmul(inputs, self.W) + tf.linalg.matmul(hidden_tm1, self.U) + self.b; # z.shape = (batch, 4 * hidden_dim)
    z0 = z[:, :self.units];
    z1 = z[:, self.units:2 * self.units];
    z2 = z[:, 2 * self.units:3 * self.units];
    z3 = z[:, 3 * self.units:];
    input_t = tf.math.sigmoid(z0); # input_t.shape = (batch, hidden_dim)
    forget_t = tf.math.sigmoid(z1); # forget_t.shape = (batch, hidden_dim)
    cell_t = forget_t * cell_tm1 + input_t * tf.math.tanh(z2); # cell_t.shape = (batch, hidden_dim)
    output_t = tf.math.sigmoid(z3); # output_t.shape = (batch, hidden_dim)
    if self.zoneout_c:
      cell_t = self.zoneout(cell_t, cell_tm1, drop_rate = self.zoneout_c);
    hidden_t = output_t *  tf.math.tanh(cell_t);
    if self.zoneout_h:
      hidden_t = self.zoneout(hidden_t, hidden_tm1, drop_rate = self.zoneout_h);
    return hidden_t, [hidden_t, cell_t];

  def get_config(self):

    config = super(ZoneoutLSTMCell, self).get_config();
    config['units'] = self.units;
    config['zoneout_h'] = self.zoneout_h;
    config['zoneout_c'] = self.zoneout_c;
    return config;

  @classmethod
  def from_config(cls, config):

    return cls(config['units'], config['zoneout_h'], config['zoneout_c']);

class TacotronDecoderCell(tf.keras.layers.Layer):

  def __init__(self, units, **kwargs):

    self.decoders = [ZoneoutLSTMCell(1024, 0.1, 0.1) for i in range(2)];

  def call(self, inputs, states):

    cell_state = states[0];
    time = states[1]
    attention = states[2];
    alignments = states[3];
    alignment_history = states[4];
    max_attentions = states[5];
    # 1) prenet
    # output shape = (batch, 256)
    results = tf.keras.layers.Dense(units = 256, activation = tf.keras.layers.ReLU())(inputs);
    results = tf.keras.layers.Dropout(rate = 0.5)(results);
    results = tf.keras.layers.Dense(units = 256, activation = tf.keras.layers.ReLU())(results);
    results = tf.keras.layers.Dropout(rate = 0.5)(results);
    # 2) lstm input
    # output shape = (batch, 256 + 1758)
    results = tf.keras.layers.Concatenate(axis = -1)([results, attention]);
    # 3) unidirectional LSTM layers
    next_cell_state = cell_state;
    for decoder in self.decoders:
      results, next_cell_state = decoder(results, next_cell_state);
    # 4) compute attention
    

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
    layer = tf.keras.layers.RNN(ZoneoutLSTMCell(enc_lstm_units), return_sequences = True),
    backward_layer = tf.keras.RNN(ZoneoutLSTMCell(enc_lstm_units), return_sequences = True, go_backwards = True),
    merge_mode = 'concat')(results);
  # 2) tacotron decoder cell
  # 2.1) prenet
  # output shape = (batch, seq_length, 256)
  results = tf.keras.layers.Dense(units = 256, activation = tf.keras.layers.ReLU())(results);
  results = tf.keras.layers.Dropout(rate = 0.5)(results);
  results = tf.keras.layers.Dense(units = 256, activation = tf.keras.layers.ReLU())(results);
  prenet_results = tf.keras.layers.Dropout(rate = 0.5)(results);
  # 2.2) location sensitive attention

if __name__ == "__main__":

  cell = ZoneoutLSTMCell(100);
  rnn = tf.keras.layers.RNN(cell);
  import numpy as np;
  a = tf.constant(np.random.normal(size = (8, 10, 200)), dtype = tf.float32);
  state = [tf.constant(np.random.normal(size = (8, 100)), dtype = tf.float32), tf.constant(np.random.normal(size = (8, 100)), dtype = tf.float32)];
  b = rnn(a, initial_state = state);
  print(b.shape)
