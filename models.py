#!/usr/bin/python3

from math import cos, pi;
import numpy as np;
import tensorflow as tf;

def PreNet(code_dim, units = 256, layers = 2, drop_rate = 0.5):

  inputs = tf.keras.Input((code_dim,)); # inputs.shape = (batch, hidden_dim)
  results = inputs;
  for i in range(layers):
    results = tf.keras.layers.Dense(units = units, activation = tf.keras.layers.ReLU(), kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3))(results);
    results = tf.keras.layers.Dropout(rate = drop_rate)(results); # results.shape = (batch, 256)
  return tf.keras.Model(inputs = inputs, outputs = results);

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

    self.W = self.add_weight(shape = (input_shape[-1], 4 * self.units), name = 'zeonout_w'); # shape = (input_dim, 4 * hidden_dim)
    self.U = self.add_weight(shape = (self.units, 4 * self.units), name = 'zoneout_u'); # shape = (hidden_dim, 4 * hidden_dim)
    self.b = self.add_weight(shape = (4 * self.units,), name = 'zoneout_b'); # shape = (4 * hidden_dim)

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
    hidden_t = output_t * tf.math.tanh(cell_t);
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

  def __init__(self, units, smoothing = False, cumulate_weights = True, synthesis_constraint = False, constraint_type = 'window', attention_win_size = 7, **kwargs):

    # layers
    self.query_layer = tf.keras.layers.Dense(units = units, kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3));
    self.memory_layer = tf.keras.layers.Dense(units = units, kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3));
    self.location_convolution = tf.keras.layers.Conv1D(filters = 32, kernel_size = 31, padding = 'same', use_bias = True, bias_initializer = tf.keras.initializers.Zeros(), kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3));
    self.location_layer = tf.keras.layers.Dense(units = units, use_bias = False, kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3));
    self.probability_fn = tf.keras.layers.Lambda(lambda e: tf.math.sigmoid(e) / tf.math.reduce_sum(tf.math.sigmoid(e), axis = -1, keepdims = True)) if smoothing else tf.keras.layers.Softmax(axis = -1);
    # params need to be serialized
    self.units = units;
    self.smoothing = smoothing;
    self.cumulate_weights = cumulate_weights;
    self.synthesis_constraint = synthesis_constraint;
    self.constraint_type = constraint_type;
    self.attention_win_size = attention_win_size;
    self.memory_intiailized = False;
    super(LocationSensitiveAttention, self).__init__(**kwargs);

  def setup_memory(self, memory):

    # NOTE: memory is neither on deep dimension nor on temporal dimension, so feed separately
    # memory = [h_{t-T},...,h_t], shape = (batch, seq_length, hidden_dim)
    self.memory = memory;
    self.memory_intiailized = True;

  def build(self, input_shape):

    self.V = self.add_weight(shape = (self.units,), initializer = tf.keras.initializers.GlorotNormal());
    self.b = self.add_weight(shape = (self.units,), initializer = tf.keras.initializers.Zeros());

  def get_initial_state(self, inputs = None, batch_size = None, dtype = None):

    tf.debugging.assert_equal(self.memory_intiailized, True, message = 'call get_initial_state after setup_memory!');
    next_state = tf.zeros((batch_size, tf.shape(self.memory)[1]), dtype = tf.float32); # next_state.shape = (batch, seq_length)
    max_attentions = tf.zeros((batch_size,), dtype = tf.int32); # max_attentions.shape = (batch,)
    return (next_state, max_attentions);

  def call(self, inputs, state):

    query = inputs; # query = s_{t-1}, shape = (batch, query_dim)
    prev_state = state[0]; # state = [a_{t-T}, ..., a_t], shape = (batch, seq_length)
    prev_max_attentions = state[1]; # max_attention.shape = (batch,)
    tf.debugging.assert_equal(prev_state.dtype, tf.float32, message = 'prev_state must be in float32 type');
    tf.debugging.assert_equal(prev_max_attentions.dtype, tf.int32, message = 'prev_max_attentions must be in int32 type');
    tf.debugging.assert_equal(self.memory_intiailized, True, message = 'memory is not set!');
    Ws = self.query_layer(query); # Ws.shape = (batch, units)
    Ws = tf.expand_dims(Ws, axis = 1); # Ws.shape = (batch, 1, units)
    keys = self.memory_layer(self.memory); # keys.shaope = (batch, seq_length, units);
    conv = self.location_convolution(tf.expand_dims(prev_state, axis = 2)); # conv.shape = (batch, seq_length, 32)
    Uconv = self.location_layer(conv); # Uconv.shape = (batch, seq_length, units)
    energy = tf.math.reduce_sum(self.V * tf.math.tanh(Ws + keys + Uconv + self.b), axis = 2); # energy.shape = (batch, seq_length)
    def constraint(energy):
      seq_length = tf.shape(energy)[-1];
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
    next_state = a_t + prev_state if self.cumulate_weights else a_t; # next_state.shape = (batch, seq_length)
    max_attentions = tf.math.argmax(a_t, -1, output_type = tf.int32); # prev_max_attentions.shape = (batch,)
    return a_t, (next_state, max_attentions);

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
def TacotronEncoder(enc_filters = 512, kernel_size = 5, enc_layers = 3, drop_rate = 0.5, enc_lstm_units = 256):

  inputs = tf.keras.Input((None, 512)); # inputs.shape = (batch, seq_length, 512)
  results = inputs;
  # 1) tacotron encoder cell
  # 1.1) convolutional layers
  # output shape = (batch, seq_length, enc_filters)
  for i in range(enc_layers):
    results = tf.keras.layers.Conv1D(filters = enc_filters, kernel_size = kernel_size, padding = 'same', activation = tf.keras.layers.ReLU(), kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3))(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.Dropout(rate = drop_rate)(results);
  # 1.2) rnn layers (will use zoneout LSTM instead when it is available in tf.keras)
  # output shape = (batch, seq_length, 2 * enc_lstm_units)
  results = tf.keras.layers.Bidirectional(
    layer = tf.keras.layers.RNN(ZoneoutLSTMCell(enc_lstm_units), return_sequences = True),
    backward_layer = tf.keras.layers.RNN(ZoneoutLSTMCell(enc_lstm_units), return_sequences = True, go_backwards = True),
    merge_mode = 'concat')(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

# decoder
class TacotronDecoderCell(tf.keras.layers.Layer):

  def __init__(self, num_mels = 80, outputs_per_step = 1, **kwargs):

    self.prenet = PreNet(code_dim = num_mels * outputs_per_step);
    self.decoder_rnn = tf.keras.layers.RNN([ZoneoutLSTMCell(1024, 0.1, 0.1) for i in range(2)], return_state = True);
    self.attention_mechanism = LocationSensitiveAttention(128, cumulate_weights = True, synthesis_constraint = False);
    self.frame_projection = tf.keras.layers.Dense(units = num_mels * outputs_per_step, kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3));
    self.stop_projection = tf.keras.layers.Dense(units = outputs_per_step, activation = tf.nn.sigmoid, kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3));
    # params need to be serialized
    self.num_mels = num_mels;
    self.outputs_per_step = outputs_per_step;
    super(TacotronDecoderCell, self).__init__(**kwargs);

  def setup_memory(self, memory):
    
    # memory.shape = (batch, seq_length, hidden_dim)
    self.attention_mechanism.setup_memory(memory);

  def get_initial_state(self, inputs = None, batch_size = None, dtype = None):
    
    # NOTE: the output shape of self.frame_projection
    c_t = tf.zeros((batch_size, self.num_mels * self.outputs_per_step), dtype = tf.float32);
    rnn_state = [[tf.zeros((batch_size, 1024), dtype = tf.float32), tf.zeros((batch_size, 1024), dtype = tf.float32)],
                 [tf.zeros((batch_size, 1024), dtype = tf.float32), tf.zeros((batch_size, 1024), dtype = tf.float32)]];
    attention_state = self.attention_mechanism.get_initial_state(batch_size = batch_size);
    return (c_t, rnn_state, attention_state);

  def call(self, inputs, states):

    prev_cell_outputs = inputs[0] # prev_cell_outputs.shape = (batch, hidden_dim)    
    c_tm1 = states[0]; # c_tm1.shape = (batch, hidden_dim)
    rnn_state = states[1]; # (hidden_state, cell_state)
    attention_state = states[2]; # (attention_state, max_attention)
    # 1) prenet
    # output shape = (batch, 256)
    results = self.prenet(prev_cell_outputs);
    # 2) lstm input
    lstm_input = tf.keras.layers.Concatenate(axis = -1)([results, c_tm1]); # lstm_input.shape = (batch, 256 + hidden_dim)
    # 3) unidirectional LSTM layers
    lstm_output, next_hidden_state, next_cell_state = self.decoder_rnn(tf.expand_dims(lstm_input, axis = 1), initial_state = rnn_state); # results.shape = (batch, 1024)
    # 4) location sensitive attention
    a_t, (attention_state, max_attentions) = self.attention_mechanism(lstm_output, attention_state); # a_t.shape = (batch, seq_length)
    c_t = tf.math.reduce_sum(tf.expand_dims(a_t, axis = -1) * self.attention_mechanism.memory, axis = 1); # expanded_a_t.shape = (batch, hidden_dim)
    # 5) compute predicted frames and predicted stop token
    projections_input = tf.concat([lstm_output, c_t], axis = -1); # projections_input.shape = (batch, 1024 + hidden_dim)
    cell_outputs = self.frame_projection(projections_input); # cell_outputs.shape = (batch, num_mels * outputs_per_step)
    stop_tokens = self.stop_projection(projections_input); # stop_outputs.shape = (batch, outputs_per_step)
    return (cell_outputs, stop_tokens), (c_t, (next_hidden_state, next_cell_state), (attention_state, max_attentions));

  def get_config(self):

    config = super(TacotronDecoderCell, self).get_config();
    config['num_mels'] = self.num_mels;
    config['outputs_per_step'] = self.outputs_per_step;

  @classmethod
  def from_config(cls, config):

    return cls(**config);

def PostNet(num_mels = 80, layers = 5, drop_rate = 0.5):

  inputs = tf.keras.Input((None, num_mels)); # inputs.shape = (batch, seq_length, num_mels)
  results = inputs;
  for i in range(layers - 1):
    results = tf.keras.layers.Conv1D(filters = 512, kernel_size = 5, padding = 'same', activation = tf.math.tanh, kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3))(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.Dropout(rate = drop_rate)(results);
  results = tf.keras.layers.Conv1D(filters = 512, kernel_size = 5, padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3))(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.Dropout(rate = drop_rate)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def CBHG(kernel_size = 8, num_mels = 80, highway_units = 128, highway_layers = 4, rnn_units = 128):

  inputs = tf.keras.Input((None, num_mels)); # inputs.shape = (batch, seq_length, num_mels)
  conv_results = list();
  for i in range(kernel_size):
    results = tf.keras.layers.Conv1D(filters = 128, kernel_size = i + 1, padding = 'same', activation = tf.keras.layers.ReLU(), kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3))(inputs);
    results = tf.keras.layers.BatchNormalization()(results);
    conv_results.append(results);
  results = tf.keras.layers.Concatenate(axis = -1)(conv_results); # results.shape = (batch, seq_length, 8 * 128)
  results = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 1, padding = 'same')(results); # results.shape = (batch, seq_length, 8 * 128)
  results = tf.keras.layers.Conv1D(filters = 256, kernel_size = 3, padding = 'same', activation = tf.keras.layers.ReLU(), kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3))(results); # results.shape = (batch, seq_length, 256)
  results = tf.keras.layers.BatchNormalization()(results); # results.shape = (batch, seq_length, 256)
  results = tf.keras.layers.Conv1D(filters = num_mels, kernel_size = 3, padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3))(results); # results.shape = (batch, seq_length, num_mels)
  results = tf.keras.layers.BatchNormalization()(results); # results.shape = (batch, seq_length, num_mels)
  results = tf.keras.layers.Add()([results, inputs]); # results.shape = (batch, seq_length, num_mels)
  if highway_units != results.shape[-1]:
    results = tf.keras.layers.Dense(units = highway_units, kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3))(results); # results.shape = (batch, seq_length, highway_units)
  for i in range(highway_layers):
    H = tf.keras.layers.Dense(units = highway_units, activation = tf.keras.layers.ReLU(), kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3))(results); # H.shape = (batch, seq_length, highway_units)
    T = tf.keras.layers.Dense(units = highway_units, activation = tf.math.sigmoid, bias_initializer = tf.keras.initializers.Constant(-1.), kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3))(results); # T.shape = (batch, seq_length, highway_units)
    results = tf.keras.layers.Lambda(lambda x: x[0] * x[1] + x[2] * (1. - x[1]))([H,T,results]); # results.shape = (batch, seq_length, highway_units)
  results = tf.keras.layers.Bidirectional(layer = tf.keras.layers.GRU(rnn_units, return_sequences = True),
                                          backward_layer = tf.keras.layers.GRU(rnn_units, return_sequences = True, go_backwards = True),
                                          merge_mode = 'concat')(results); # results.shape = (batch, seq_length, 2 * rnn_units)
  return tf.keras.Model(inputs = inputs, outputs = results);

class Tacotron2(tf.keras.Model):

  def __init__(self, num_mels = 80, outputs_per_step = 1):

    super(Tacotron2, self).__init__();
    self.encoder = TacotronEncoder();
    self.decoder_cell = TacotronDecoderCell(num_mels = num_mels, outputs_per_step = outputs_per_step);
    self.postnet = PostNet(num_mels = num_mels);
    self.frame_projection = tf.keras.layers.Dense(units = num_mels, kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3));
    self.frame_projection2 = tf.keras.layers.Dense(units = num_mels, kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3));
    self.cbhg = CBHG(num_mels = num_mels);

  def ratio(self, iterations):

    alpha = 0;
    global_step = iterations - 10000;
    global_step = min(global_step, 40000);
    cosine_decay = 0.5 * (1 + cos(pi * global_step / 40000));
    decayed = (1 - alpha) * cosine_decay + alpha;
    decayed_learning_rate = 1. * decayed;
    return decayed_learning_rate;

  def call(self, inputs):
    
    # inputs.shape = (batch, seq_length, 512)
    code = self.encoder(inputs);
    self.decoder_cell.setup_memory(code);
    state = self.decoder_cell.get_initial_state(batch_size = tf.shape(inputs)[0]);
    output = (tf.zeros((tf.shape(inputs)[0], self.decoder_cell.num_mels * self.decoder_cell.outputs_per_step)),
              tf.zeros((tf.shape(inputs)[0], self.decoder_cell.outputs_per_step))); # initial frame is zero
    outputs = list();
    while True:
      output, state = self.decoder_cell(output, state); # output.shape = (batch, num_mels, outputs_per_step)
      outputs.append(output[0]);
      finished = tf.cast(tf.math.round(output[1]), dtype = tf.bool); # finished.shape = (batch, outputs_per_step)
      finished = tf.math.reduce_any(finished);
      if finished == True: break;
      if len(outputs) > 10000: break;
    results = tf.keras.layers.Concatenate(axis = 1)(outputs); # results.shape = (batch, output_length, outputs_per_step)
    decoder_outputs = tf.keras.layers.Reshape((None, self.decoder_cell.num_mels))(results); # results.shape = (batch, output_length, num_mels)
    decoder_outputs = tf.clip_by_value(decoder_outputs, clip_value_min = -4. - 0.1, clip_value_max = 4.); # results.shape = (batch, output_length, num_mels)
    results = self.postnet(decoder_outputs); # results.shape = (batch, output_length, 512)
    results = self.frame_projection(results); # results.shape = (batch, output_length, num_mels)
    mel_outputs = tf.keras.layers.Add()([results, decoder_outputs]); # mel_outputs.shape = (batch, output_length, num_mels)
    mel_outputs = tf.clip_by_value(mel_outputs, clip_value_min = -4. - 0.1, clip_value_max = 4.); # mel_outputs.shape = (batch, output_length, num_mels)
    return mel_outputs;

  def loss(self, inputs, labels, feed_mode = 'groundtruth', iterations = None, sample_rate = 22050, num_freq = 1025):

    # inputs.shape = (batch, seq_length)
    # labels.shape = (batch, label_length, num_mels)
    # feed_mode:
    # 1) ground truth: always feed ground truth
    # 2) eval: always feed prediction of previous time
    # 3) scheduled: feed ground truth at first and gradually use feed prediction of previous time
    if feed_model == 'scheduled': assert iterations is not None;
    ratio = 1. if feed_mode == 'groundtruth' else (0. if feed_mode == 'eval' else (self.ratio(iterations) if feed_model == 'scheduled' else None));
    if ratio is None: raise Exception("feed mode must be among 'groundtruth', 'eval' and 'scheduled'");
    # follow implement of TacoTrainingHelper
    code = self.encoder(inputs);
    state = self.decoder_cell.get_initial_state(batch_size = tf.shape(inputs)[0]);
    output = (tf.zeros((tf.shape(inputs)[0], self.decoder_cell.num_mels, self.decoder_cell.outputs_per_step)),
              tf.zeros((tf.shape(inputs)[0], self.decoder_cell.outputs_per_step))); # initial frame is zero
    outputs = list();
    stop_tokens = list();
    for i in range(tf.shape(labels)[1] - 1):
      output = (tf.cond(tf.less(tf.random_uniform((), minval = 0, max_val = 1, dtype = tf.float32), ratio),
                        lambda: labels[:,i,:], lambda: output[0] if i == 0 else outputs[-1]), output[1]);
      output, state = self.decoder_cell(output, state); # output.shape = (batch, num_mels, outputs_per_step)
      outputs.append(output[0]); # output[0].shape = (batch, num_mels, outputs_per_step)
      stop_tokens.append(output[1]); # output[1].shape = (batch, output_per_step)
    assert(len(outputs) == tf.shape(labels)[1] - 1);
    results = tf.keras.layers.Concatenate(axis = 1)(outputs); # results.shape = (batch, label_length - 1, outputs_per_step)
    decoder_outputs = tf.keras.layers.Reshape((None, self.decoder_cell.num_mels))(results); # results.shape = (batch, label_length - 1, num_mels)
    decoder_outputs = tf.clip_by_value(decoder_outputs, clip_value_min = -4. - 0.1, clip_value_max = 4.); # results.shape = (batch, label_length - 1, num_mels)
    stop_tokens = tf.keras.layers.Concatenate(axis = 1)(stop_tokens); # stop_tokens.shape = (batch, outpuut_per_step * (label_length - 1))
    results = self.postnet(decoder_outputs); # results.shape = (batch, label_length - 1, 512)
    results = self.frame_projection(results); # results.shape = (batch, label_length - 1, num_mels)
    mel_outputs = tf.keras.layers.Add()([results, decoder_outputs]); # mel_outputs.shape = (batch, label_length - 1, num_mels)
    mel_outputs = tf.clip_by_value(mel_outputs, clip_value_min = -4. - 0.1, clip_value_max = 4.); # mel_outputs.shape = (batch, label_length - 1, num_mels)
    # post condition
    post_cbhg = self.cbhg(mel_outputs); # post_cbhg.shape = (batch, label_length - 1, 2 * rnn_units)
    linear_specs_projection = self.frame_projection2(post_cbhg); # linear_specs_projection.shape = (batch, label_length - 1, num_mels)
    # get loss
    regression_loss = tf.keras.losses.MSE(labels[:,1:,:], decoder_outputs) + tf.keras.losses.MSE(labels[:,1:,:], mel_outputs);
    stop_tokens_gt = tf.keras.layers.Concatenate(axis = -1)([tf.zeros((tf.shape(inputs)[0], tf.shape(labels)[1] - 2), dtype = tf.float32), 
                                                             tf.ones((tf.shape(inputs)[0], 1), dtype = tf.float32)]); # stop_tokens_gt.shape = (batch, label_length - 1)
    classification_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)(stop_tokens_gt, stop_tokens);
    l1 = tf.math.abs(labels[:,1:,:] - linear_specs_projection);
    n_priority_freq = int(2000 / (sample_rate * 0.5) * num_freq);
    linear_loss = 0.5 * tf.math.reduce_mean(l1) + 0.5 * tf.math.reduce_mean(l1[:,:,n_priority_freq]);
    return regression_loss + classification_loss + linear_loss;

if __name__ == "__main__":

  # 0) prenet
  prenet = PreNet(code_dim = 512);
  a = tf.constant(np.random.normal(size = (8,512)), dtype = tf.float32);
  b = prenet(a);
  print(b.shape);
  prenet.save('prenet.h5');
  # 1) zoneout lstm
  cell = ZoneoutLSTMCell(100);
  rnn = tf.keras.layers.RNN(cell);
  a = tf.constant(np.random.normal(size = (8, 10, 200)), dtype = tf.float32);
  state = [tf.constant(np.random.normal(size = (8, 100)), dtype = tf.float32), tf.constant(np.random.normal(size = (8, 100)), dtype = tf.float32)];
  b = rnn(a, initial_state = state);
  print(b.shape)
  # 2) location sensitive attention
  lsa = LocationSensitiveAttention(100, synthesis_constraint = True);
  lsa.setup_memory(tf.zeros((8,10,32)));
  state = lsa.get_initial_state(batch_size = 8);
  query = tf.constant(np.random.normal(size = (8, 64)));
  a_t, (state, max_attention) = lsa(query, state);
  print(a_t.shape)
  # 3) tacotron encoder
  encoder = TacotronEncoder();
  a = tf.constant(np.random.normal(size = (8, 100, 512)), dtype = tf.float32);
  b = encoder(a);
  print(b.shape);
  encoder.save('tacotronencoder.h5');
  # 4) TacotronDecoderCell
  decoder = TacotronDecoderCell();
  decoder.setup_memory(a);
  state = decoder.get_initial_state(batch_size = 8);
  inputs = (tf.zeros((8, 80)), tf.zeros((8, 1)));
  outputs = decoder(inputs, state);
  print(outputs[0][0].shape)
  print(outputs[0][1].shape)
  # 5) PostNet
  postnet = PostNet();
  a = tf.constant(np.random.normal(size = (8, 100, 80)));
  b = postnet(a);
  print(b.shape);
  postnet.save('postnet.h5');
  # 6) CBHG
  cbhg = CBHG();
  a = tf.constant(np.random.normal(size = (8, 100, 80)));
  b = cbhg(a);
  print(b.shape);
  cbhg.save('cbhg.h5');
  # 7) Tacotron2
  tacotron2 = Tacotron2();
  a = tf.constant(np.random.normal(size = (8, 100, 512)));
  b = tacotron2(a);
  print(b.shape);
  tacotron2.save_weights('tacotron2.h5');
