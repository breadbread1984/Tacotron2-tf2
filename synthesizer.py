#!/usr/bin/python3

import librosa;
import pyaudio;
import numpy as np;
import tensorflow as tf;
from models import Tacotron2;
from tokenizer import Tokenizer;

class Synthesizer(object):

  def __init__(self,):

    super(Synthesizer, self).__init__();
    self.tts = Tacotron2();
    self.tokenizer = Tokenizer();

  def denormalize(self, mel_outputs):

    outputs = (tf.clip_by_value(mel_outputs, clip_value_min = -4., clip_value_max = 4.) + 4.) * -100. / (2 * 4.) + 100.;
    return outputs;

  def db_to_amp(self, db):
    # decibel to ampere
    return tf.pow(tf.ones_like(db) * 10., db * 0.05);

  def griffin_lim(self, s):
      
    s = tf.expand_dims(s, axis = 0);
    s_complex = tf.cast(s, dtype = tf.complex64);
    y = tf.signal.inverse_stft(s_complex, 1100, 275, 2048);
    for i in range(60):
      est = tf.signal.stft(y, 1100, 275, 2048);
      angles = est / tf.cast(tf.math.maximum(1e-8, tf.math.abs(est)), tf.complex64);
      y = tf.signal.inverse_stft(s_complex * angles, 1100, 275, 2048);
    s = tf.squeeze(y, axis = 0);
    return s;

  def inv_linear_spectrogram(self, linear_outputs):

    d = self.denormalize(linear_outputs);
    s = tf.pow(self.db_to_amp(d + 20), (1/2.));
    return self.griffin_lim(tf.pow(s, 1.5));

  def inv_mel_spectrogram(self, mel_outputs):

    d = self.denormalize(mel_outputs);
    s = tf.pow(self.db_to_amp(d + 20), (1/2.));
    # mel to linear
    inv_mel_basis = tf.linalg.pinv(librosa.filters.mel(22050, 2048, n_mels = 80, fmin = 55, fmax = 7600));
    s = tf.transpose(tf.math.maximum(1e-10, tf.linalg.matmul(tf.cast(inv_mel_basis, tf.float32), tf.transpose(s, (1,0)))), (1,0));
    return self.griffin_lim(tf.pow(s, 1.5));

  def call(self, text):

    tokens = self.tokenizer.tokenize(text); # tokens.shape = (seq_length,)
    tokens = tf.expand_dims(tokens, axis = 0); # tokens.shape = (1, seq_length)
    embedding = tf.keras.layers.Embedding(self.tokenizer.size(), 512)(tokens); # embedding.shape = (1, token_length, 512)
    mel_outputs, linear_outputs = self.tts(embedding); # mel_outputs.shape = (1, token_length, 80)
                                                       # linear_outputs.shape = (1, token_length, 512)
    mel = self.inv_mel_spectrogram(mel_outputs);
    linear = self.inv_linear_spectrogram(linear_outputs);
    # TODO
    return mel_outputs;
