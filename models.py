#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_addons as tfa;

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

# LocationSensitiveAttention is an implement of attention algorithm introduced in paper
# Attention-Based Models for Speech Recognition
class LocationSensitiveAttention(tfa.seq2seq.BahdanauAttention):

  def __init__(self, units, memory, memory_sequence_length = None, mask_encoder = True, smoothing = False, cumulate_weights = True, **kwargs):

    # memory = [h_{t-T},...,h_t], shape = (batch, seq_length, hidden_dim)
    super(LocationSensitiveAttention, self).__init__(units = units, memory = memory, memory_sequence_length = memory_sequence_length if mask_encoder else None, normalize = True, **kwargs);
    self.location_convolution = tf.keras.layers.Conv1D(filters = 32, kernel_size = 31, padding = 'same', use_bias = True, bias_initializer = tf.zeros_initializer());
    self.location_layer = tf.keras.layers.Dense(units = units, use_bias = False);
    # params need to be serialized
    self.mask_encoder = mask_encoder;
    self.smoothing = smoothing;
    self.cumulate_weights = cumulate_weights;

  def smoothing_normalization(self, e, _):
    return tf.math.sigmoid(e) / tf.math.reduce_sum(tf.math.sigmoid(e), axis = -1, keepdims = True);

  def build(self, input_shape):

    super(LocationSensitiveAttention, self).build(input_shape);
    self.V = self.add_weight(shape = (self.units,), initializer = tf.keras.initializers.GlorotNormal());
    self.b = self.add_weight(shape = (self.units,), initializer = tf.keras.initializers.Zeros());

  def call(self, inputs):

    s_tm1 = inputs[0]; # query
    a_tm1 = inputs[1]; # state
    # s_tm1 is s_{t-1}, shape = (batch, query_dim)
    # a_tm1 is a_{t-1}, shape = (batch, seq_length)
    Ws = self.query_layer(s_tm1); # Ws.shape = (batch, units)
    Ws = tf.expand_dims(Ws, axis = 1); # Ws.shape = (batch, 1, units)
    a_tm1 = tf.expand_dims(a_tm1, axis = 2); # a_tm1.shape = (batch, seq_length, 1)
    conv = self.location_convolution(a_tm1); # conv.shape = (batch, seq_length, 32)
    Uconv = self.location_layer(conv); # Uconv.shape = (batch, seq_length, units)
    # NOTE: keys.shape = V_{units x hidden_dim} cdot [h_{t-T}, ... , h_t] = (batch, seq_length, units)
    energy = tf.math.reduce_sum(self.V * tf.math.tanh(Ws + self.keys + Uconv + self.b), axis = 2); # energy.shape = (batch, seq_length)
    # NOTE: self.probability_fn uses smoothed probability fn, argument a_tm1 is not used
    a_t = self.probability_fn(energy, a_tm1) if self.smoothing else tf.keras.layers.Softmax()(energy); # a_t.shape = (batch, seq_length)
    max_attentions = tf.math.argmax(a_t, -1, output_type = tf.int32); # max_attention.shape = ()
    next_state = a_t + a_tm1 if self.cumulate_weights else a_t;
    return a_t, next_state;

  def get_config(self):

    config = super(LocationSensitiveAttention, self).get_config();
    config['mask_encoder'] = self.mask_encoder;
    config['smoothing'] = self.smoothing;
    config['cumulate_weights'] = self.cumulate_weights;

  @classmethod
  def from_config(cls, config):

    return cls(**config);

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
    # 4) location sensitive attention
    results = tf.keras.layers.Conv1D(filters = 32, kernel_size = 31, padding = 'same', )

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
  # TODO

if __name__ == "__main__":

  cell = ZoneoutLSTMCell(100);
  rnn = tf.keras.layers.RNN(cell);
  import numpy as np;
  a = tf.constant(np.random.normal(size = (8, 10, 200)), dtype = tf.float32);
  state = [tf.constant(np.random.normal(size = (8, 100)), dtype = tf.float32), tf.constant(np.random.normal(size = (8, 100)), dtype = tf.float32)];
  b = rnn(a, initial_state = state);
  print(b.shape)
  lsa = LocationSensitiveAttention(100, tf.zeros(8, 10, 32));
  s_t = tf.constant(np.random.normal(size = (8, 64)));
  a_tm1 = tf.constant(np.random.normal(size = (8, 10)));
  a_t, next_state = lsa([s_t, a_tm1]);
  print(a_t.shape)
  print(next_state.shape)
