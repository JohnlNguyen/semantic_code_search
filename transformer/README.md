# Generic transformer implementation
This directory includes a simple TF-2 Eager compatible transformer model (and the related util code and config Yaml). To instantiate, assuming you have the config loaded as in the homework code, simply import it and call Transformer(config["model"], vocab_dim).

Some implementation details:
- This model returns the final hidden states (to keep it general); you can use this to predict a token by calling its "predict" method with those states.
- The model takes three optional arguments: mask, key_states and training. I'll explain the first two under "uses" below; the last one should be set to True in training and False otherwise so that it does not drop-out at test time.
- A lot of things are configurable in config.yml and all the current configuration is based on the paper. You are welcome to add additional configuration to it though, if needed for more advanced usage (best practice evolves over time).
- The "warmup" parameter in config is related to slowly increasing the learning rate initially. It can be reduced by quite a bit (e.g. from 4,000 to 100) if you want faster initial training (which is good for quick prototyping), though for very large datasets and many training epochs I would leave it as it is or even raise it to 16K (as is now common practice for very big training tasks).

There are several uses:
- For generative language modeling (as in the homework), call the model with your indices (typically shifted by one, as in the homework) and a sequence mask to avoid attention on future tokens. A function to make this mask is provided as well, use the get_sequence_mask function. Example:
  ```
  states = model(indices[:, :-1], mask=model.get_sequence_mask(indices.shape[-1] - 1), training=training)
  preds = model.predict(states)
  ```
- For sequence tagging (e.g. part-of-speech tagging, type inference), we don't mask future tokens, but we do mask padding tokens. To do so, simply pass the "mask" that comes from your batcher instead of using the get_sequence_mask function.
- For translation, run two different transformer models in sequence: one to encode, passing just a sequence padding mask, and one to decode, passing both the encoder states and a sequence-"future" mask (as in the generative language modeling task).

A note on learning rate decay: I didn't include the actual learning rate calculation in here before, so here it is:

```
def compute_learning_rate(curr_batch_number):
	warmup_steps = config["training"]["warmup"]
	learning_rate = config["training"]["lr"] * config["model"]["hidden_dim"]**-0.5
	learning_rate *= tf.minimum(1.0, curr_batch_number / warmup_steps)
	learning_rate *= tf.math.rsqrt(tf.cast(tf.maximum(curr_batch_number, warmup_steps), "float32"))
	return learning_rate
```
To use, keep track of the overall (not per-epoch!) minibatch index in your training code and, before calling the model, increment it and call:
`learning_rate.assign(compute_learning_rate(base_lr, config["training"]["warmup"], total_batches))`
Here I assume that your learning_rate is indeed declared as a `tf.Variable`, as in the class code. Note that you must use `assign` rather than `learning_rate = ...` to actually replace the learning rate in the optimizer!