import sys
import os
import math
import random
import yaml
import argparse

import numpy as np
import tensorflow as tf

random.seed(41)
from data_reader import DataReader
config = yaml.safe_load(open("config.yml"))

log_2_e = 1.44269504089 # Constant to convert to binary entropies

class AttentionLayer(tf.keras.layers.Layer):
	def __init__(self, attn_dim, num_heads=None):
		super(AttentionLayer, self).__init__()
		self.attn_dim = attn_dim
		self.num_heads = 1 if num_heads == None else num_heads
		self.attn_keys = tf.keras.layers.Dense(self.attn_dim * self.num_heads, activation="tanh")
		self.attn_query = tf.keras.layers.Dense(self.attn_dim * self.num_heads, activation="tanh")
		self.attn_values = tf.keras.layers.Dense(self.attn_dim * self.num_heads, activation="tanh")
	
	def build(self, input_shape):
		self.weight_out = None if self.num_heads == 1 else tf.keras.layers.Dense(input_shape[-1], activation="tanh")
	
	def call(self, states):
		batch_dim = states.shape[0]
		seq_len = states.shape[1]
		
		# Util function to move the num_heads dimension to the start, allowing easier computation
		def reshape_for_heads(value):
			value = tf.reshape(value, [batch_dim, seq_len, self.num_heads, self.attn_dim])
			value = tf.transpose(value, [0, 2, 1, 3])
			return value
		
		# Compute query, keys, and values
		query = self.attn_query(states)
		keys = self.attn_keys(states)
		values = self.attn_values(states)
		if self.num_heads > 1: # If using more than one head, reshape to [Batch, Heads, Seq_len, Dim]
			keys = reshape_for_heads(keys)
			query = reshape_for_heads(query)
			values = reshape_for_heads(values)
		
		# Scaled dot-product attention: multiply queries and keys, and normalize by sqrt of dim
		alpha = tf.matmul(query, keys, transpose_b=True)
		alpha /= tf.math.sqrt(tf.cast(self.attn_dim, "float32"))
		
		# Mask out future values to avoid attending to those
		alpha_mask = tf.sequence_mask(lengths=list(range(1, seq_len + 1)), maxlen=seq_len, dtype=tf.float32)
		alpha = alpha * alpha_mask + (1.0 - alpha_mask) * tf.float32.min # Set impermissable values to -inf
		alpha = tf.nn.softmax(alpha) # Take softmax afterwards
		
		# Multiply alpha (probability distribution over other tokens) by values to get responses to query
		context = tf.matmul(alpha, values)
		
		# In case of more than one head, reshape result back again
		if self.num_heads > 1:
			context = tf.transpose(context, [0, 2, 1, 3])
			context = tf.reshape(context, [batch_dim, seq_len, self.num_heads*self.attn_dim])
		
		# Apply one more transform to ensure we have the right shape.
		# If we just used "states" instead of "values" when computing "context" (as is typical for simple attention), this wouldn't be necessary.
		return self.weight_out(context)

class WordModel(tf.keras.Model):
	def __init__(self, embed_dim, hidden_dim, num_layers, vocab_dim):
		super(WordModel, self).__init__()
		random_init = tf.random_normal_initializer(stddev=0.1)
		self.embed = tf.Variable(random_init([vocab_dim, embed_dim]), dtype=tf.float32)
		self.rnns = [tf.keras.layers.GRU(hidden_dim, return_sequences=True) for _ in range(num_layers)]
		self.attention = [AttentionLayer(64, 1) for _ in range(num_layers)]
		self.bn = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
		self.do = [tf.keras.layers.Dropout(0.5) for _ in range(num_layers)]
		self.project = tf.keras.layers.Dense(vocab_dim)
	
	# Very basic RNN-based language model: embed inputs, encode through several layers and project back to vocabulary
	def call(self, indices, training=True):
		states = tf.nn.embedding_lookup(self.embed, indices)
		for ix, rnn in enumerate(self.rnns):
			new_states = rnn(states)
			new_states = tf.concat([new_states, self.attention[ix](states)], -1)
			states = new_states
			states = self.bn[ix](states, training=training)
			states = self.do[ix](states, training=training)
		preds = self.project(states)
		return preds

def eval(model, data):
	mbs = 0
	count = 0
	entropy = 0.0
	for indices, masks in data.batcher(data.valid_data, is_training=False):
		mbs += 1
		samples = int(tf.reduce_sum(masks[:, 1:]).numpy())
		count += samples
		preds = model(indices[:, :-1], training=False)
		loss = masked_ce_loss(indices, masks, preds)
		entropy += log_2_e * float(samples*loss.numpy())
	entropy = entropy / count
	return entropy, count

# Compute cross-entropy loss, making sure not to include "masked" padding tokens
def masked_ce_loss(indices, masks, preds):
	samples = tf.reduce_sum(masks[:, 1:])
	loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(indices[:, 1:], preds.shape[-1]), logits=preds)
	loss *= masks[:, 1:]
	loss = tf.reduce_sum(loss) / samples
	return loss

def train(model, data):
	# Declare the learning rate as a variable to include it in the saved state
	learning_rate = tf.Variable(config["training"]["lr"], name="learning_rate")
	optimizer = tf.keras.optimizers.Adam(learning_rate)
	
	# Create a checkpointer to let us save models in progress
	checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
	# Restore previous checkpoint (if any) -- tentatively out-commented, but feel free to use
	#if os.path.exists("checkpoints"):
	#	checkpoint.restore(tf.train.latest_checkpoint("checkpoints/"))
	
	is_first = True
	for epoch in range(config["training"]["num_epochs"]):
		print("Epoch:", epoch + 1)
		mbs = 0
		words = 0
		avg_loss = 0
		# Batcher returns a square index array and a binary mask indicating which words are padding (0) and real (1)
		for indices, masks in data.batcher(data.train_data):
			mbs += 1
			samples = tf.reduce_sum(masks[:, 1:])
			words += int(samples.numpy())
			
			# Run through one batch to init variables
			if is_first:
				model(indices[:, :-1])
				is_first = False
			
			# Compute loss in scope of gradient-tape (can also use implicit gradients)
			with tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(model.trainable_variables)
				preds = model(indices[:, :-1])
				loss = masked_ce_loss(indices, masks, preds)
			
			# Collect gradients, clip and apply
			grads = tape.gradient(loss, model.trainable_variables)
			grads, _ = tf.clip_by_global_norm(grads, 0.25)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			
			# Update average loss and print if applicable
			avg_loss += log_2_e * loss
			if mbs % config["training"]["print_freq"] == 0:
				avg_loss = avg_loss.numpy()/config["training"]["print_freq"]
				print("MB: {0}: words: {1}, entropy: {2:.3f}".format(mbs, words, avg_loss))
				avg_loss = 0.0
		
		# Run a validation pass at the end of every epoch
		entropy, count = eval(model, data)
		print("Validation: tokens: {0}, entropy: {1:.3f}, perplexity: {2:.3f}".format(count, entropy, 0.0 if entropy > 100 else math.pow(2, entropy)))
		checkpoint.save("checkpoints/ckpt") # Save a checkpoint for this epoch (may choose to only do this if validation entropy improved)

def main():
	# Extract arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("train_data", help="Path to training data")
	ap.add_argument("-v", "--valid_data", required=False, help="(optional) Path to held-out (validation) data")
	ap.add_argument("-t", "--test_data", required=False, help="(optional) Path to test data")
	args = ap.parse_args()
	print("Using configuration:", config)
	data = DataReader(args.train_data, args.valid_data, args.test_data)
	model = WordModel(config["model"]["embed_dim"], config["model"]["hidden_dim"], config["model"]["num_layers"], data.vocab_dim)
	random.shuffle(data.valid_data) # Shuffle just once
	train(model, data)

if __name__ == '__main__':
	main()
