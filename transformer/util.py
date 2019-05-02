import numpy as np
import tensorflow as tf

# Based on https://github.com/DongjunLee/transformer-tensorflow/blob/master/transformer/attention.py
pos_cache = None
def positional_encoding(dim, sentence_length, dtype=tf.float32):
	global pos_cache
	if pos_cache != None and pos_cache[1] == dim:
		if pos_cache[0] == sentence_length: return pos_cache[2]
		elif pos_cache[0] > sentence_length: return pos_cache[2][:sentence_length]
		else: pos_cache = None
	encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
	encoded_vec[::2] = np.sin(encoded_vec[::2])
	encoded_vec[1::2] = np.cos(encoded_vec[1::2])
	pos_enc = tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)
	pos_cache = (sentence_length, dim, pos_enc)
	return pos_enc

def tensor_matrix_mul(t, m):
	return tf.reshape(tf.reshape(t, [-1, t.shape[-1]]) @ m, [-1, t.shape[1], m.shape[-1]])

def vec_mat_mul(vector, matrix):
	return tf.multiply(tf.expand_dims(vector, -1), matrix)
