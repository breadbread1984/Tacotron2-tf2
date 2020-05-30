#!/usr/bin/python3

import tensorflow as tf;

# ZoneoutLSTM is an implement of over-fitting free LSTM introduced in a paper
# Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations
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

    return cls(**config);

# LocationSensitiveAttention is an implement of attention algorithm introduced in paper
# Attention-Based Models for Speech Recognition
class LocationSensitiveAttention(tf.keras.layers.Layer):

  def __init__(self, units, smoothing = False, cumulate_weights = True, synthesis_constraint = False, constraint_type = 'window', attention_win_size = 7):

    super(LocationSensitiveAttention, self).__init__();
    # layers
    self.query_layer = tf.keras.layers.Dense(units = units);
    self.memory_layer = tf.keras.layers.Dense(units = units);
    self.location_convolution = tf.keras.layers.Conv1D(filters = 32, kernel_size = 31, padding = 'same', use_bias = True, bias_initializer = tf.zeros_initializer());
    self.location_layer = tf.keras.layers.Dense(units = units, use_bias = False);
    self.probability_fn = tf.keras.layers.Lambda(lambda e: tf.math.sigmoid(e) / tf.math.reduce_sum(tf.math.sigmoid(e), axis = -1, keepdims = True)) if smoothing else tf.keras.layers.Softmax(axis = -1);
    # params need to be serialized
    self.units = units;
    self.smoothing = smoothing;
    self.cumulate_weights = cumulate_weights;
    self.synthesis_constraint = synthesis_constraint;
    self.constraint_type = constraint_type;
    self.attention_win_size = attention_win_size;
    self.memory_intiailized = False;
    
  def setup_memory(self, memory):

    # memory = [h_{t-T},...,h_t], shape = (batch, seq_length, hidden_dim)
    self.memory = memory;
    self.memory_intiailized = True;

  def build(self, input_shape):

    self.V = self.add_weight(shape = (self.units,), initializer = tf.keras.initializers.GlorotNormal());
    self.b = self.add_weight(shape = (self.units,), initializer = tf.keras.initializers.Zeros());

  def call(self, inputs, state):

    a_tm1 = inputs; # inputs = [a_{t-T}, ..., a_t], shape = (batch, seq_length)
    s_tm1 = state; # state = s_{t-1}, shape = (batch, query_dim)
    tf.debugging.assert_equal(self.memory_intiailized, True, message = 'memory is not set!');
    Ws = self.query_layer(s_tm1); # Ws.shape = (batch, units)
    Ws = tf.expand_dims(Ws, axis = 1); # Ws.shape = (batch, 1, units)
    keys = self.memory_layer(self.memory); # keys.shaope = (batch, seq_length, units);
    conv = self.location_convolution(tf.expand_dims(a_tm1, axis = 2)); # conv.shape = (batch, seq_length, 32)
    Uconv = self.location_layer(conv); # Uconv.shape = (batch, seq_length, units)
    energy = tf.math.reduce_sum(self.V * tf.math.tanh(Ws + keys + Uconv + self.b), axis = 2); # energy.shape = (batch, seq_length)
    def constraint(energy):
      seq_length = tf.shape(energy)[-1];
      prev_max_attentions = tf.math.argmax(a_tm1, -1, output_type = tf.int32); # prev_max_attentions.shape = (batch,)
      if self.constraint_type == 'monotonic':
        key_masks = tf.sequence_mask(prev_max_attentions, seq_length);
        reverse_masks = tf.reverse(tf.sequence_mask(seq_length - self.attention_win_size - prev_max_attentions, seq_length), axis = [-1]);
      elif self.constraint_type == 'window':
        key_masks = tf.sequence_mask(prev_max_attentions - (self.attention_win_size // 2 + (self.attention_win_size % 2 != 0)), seq_length);
        reverse_masks = tf.reverse(tf.sequence_mask(seq_length - (self.attention_win_size // 2) - prev_max_attentions, seq_length), axis = [-1]);
      else:
        raise Exception('unknonw type of synthesis constraint!');
      masks = tf.math.logical_or(key_masks, reverse_masks);
      paddings = tf.ones_like(energy) * (-2 ** 32 + 1);
      energy = tf.where(tf.equal(masks, False), energy, paddings);
      return energy;
    if self.synthesis_constraint:
      energy = tf.keras.backend.in_train_phase(energy, constraint(energy));
    a_t = self.probability_fn(energy); # a_t.shape = (batch, seq_length)
    next_state = a_t + a_tm1 if self.cumulate_weights else a_t;
    return a_t, next_state;

  def get_config(self):

    config = super(LocationSensitiveAttention, self).get_config();
    config['units'] = self.units;
    config['smoothing'] = self.smoothing;
    config['cumulate_weights'] = self.cumulate_weights;
    config['synthesis_constraint'] = self.synthesis_constraint;
    config['constraint_type'] = self.constraint_type;
    config['attention_win_size'] = self.attention_win_size;

  @classmethod
  def from_config(cls, config):

    return cls(**config);

# encoder
def TacotronEncoderCell(enc_filters = 512, kernel_size = 5, enc_layers = 3, drop_rate = 0.5, enc_lstm_units = 256):

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
  return tf.keras.Model(inputs = inputs, outputs = results);

# decoder
class TacotronDecoderCell(tf.keras.layers.Layer):

  def __init__(self, num_mels = 80, outputs_per_step = 1, **kwargs):

    self.decoder_rnn = tf.keras.layers.RNN([ZoneoutLSTMCell(1024, 0.1, 0.1) for i in range(2)], return_state = True);
    self.attention_mechanism = LocationSensitiveAttention(128, smoothing = False, cumulate_weights = True, synthesis_constraint = False);
    self.frame_projection = tf.keras.layers.Dense(units = num_mels * outputs_per_step);
    self.stop_projection = tf.keras.layers.Dense(units = outputs_per_step, activation = tf.nn.sigmoid);
    # params need to be serialized
    self.num_mels = num_mels;
    self.outputs_per_step = outputs_per_step;
    self.memory_intiailized = False;

  def setup_memory(self, memory):
    
    # memory.shape = (batch, seq_length, hidden_dim)
    self.memory = memory;
    self.memory_intiailized = True;

  def call(self, inputs, states):

    # inputs.shape = (batch, hidden_dim)
    tf.debugging.assert_equal(self.memory_intiailized, True, message = 'memory is not set!');
    
    cell_state = states[0];
    c_tm1 = states[2]; # c_tm1.shape = (batch, hidden_dim)
    attention = states[3];
    # 1) prenet
    # output shape = (batch, 256)
    results = tf.keras.layers.Dense(units = 256, activation = tf.keras.layers.ReLU())(inputs);
    results = tf.keras.layers.Dropout(rate = 0.5)(results);
    results = tf.keras.layers.Dense(units = 256, activation = tf.keras.layers.ReLU())(results);
    results = tf.keras.layers.Dropout(rate = 0.5)(results);
    # 2) lstm input
    # output shape = (batch, 256 + hidden_dim)
    results = tf.keras.layers.Concatenate(axis = -1)([results, c_tm1]);
    # 3) unidirectional LSTM layers
    results, next_cell_state = self.decoder_rnn(results, initial_state = cell_state); # results.shape = (batch, 1024)
    # 4) location sensitive attention
    self.attention_mechanism.setup_memory(self.memory);
    a_t, next_attention = self.attention_mechanism(results, attention); # a_t.shape = (batch, seq_length)
    c_t = tf.math.reduce_sum(tf.expand_dims(a_t, axis = -1) * self.memory, axis = 1); # expanded_a_t.shape = (batch, hidden_dim)
    projections_input = tf.concat([results, c_t], axis = -1); # projections_input.shape = (batch, 1024 + hidden_dim)
    cell_outputs = self.frame_projection(projections_input); # cell_outputs.shape = (batch, num_mels * outputs_per_step)
    stop_tokens = self.stop_projection(projections_input); # stop_outputs.shape = (batch, outputs_per_step)
    return (cell_outputs, stop_tokens), (next_cell_state, c_t, next_attention);

  def get_config(self):

    config = super(TacotronDecoderCell, self).get_config();
    config['num_mels'] = self.num_mels;
    config['outputs_per_step'] = self.outputs_per_step;
    
  @classmethod
  def from_config(cls, config):

    cls(**config);

if __name__ == "__main__":

  cell = ZoneoutLSTMCell(100);
  rnn = tf.keras.layers.RNN(cell);
  import numpy as np;
  a = tf.constant(np.random.normal(size = (8, 10, 200)), dtype = tf.float32);
  state = [tf.constant(np.random.normal(size = (8, 100)), dtype = tf.float32), tf.constant(np.random.normal(size = (8, 100)), dtype = tf.float32)];
  b = rnn(a, initial_state = state);
  print(b.shape)
  lsa = LocationSensitiveAttention(100,synthesis_constraint = True);
  lsa.setup_memory(tf.zeros((8,10,32)));
  s_t = tf.constant(np.random.normal(size = (8, 64)));
  a_tm1 = tf.constant(np.random.normal(size = (8, 10)));
  a_t, next_state = lsa(a_tm1, s_t);
  print(a_t.shape)
  print(next_state.shape)
