# Seq2Seq-TensorFlow
This file contains the Seq2Seq model implemented by Tensorflow.

The code is very simple to understand Seq2seq. The model learns a pair of sentences (Hello World -> How are you).

If you want to learn more sentence pairs, adjust 'input_string', 'target_string' variables.

The code was tested in the '1.0.0-rc0' version.


## Data and parameter setting
```python
vocab_size=256 # # of ASCII Code
target_vocab_size=vocab_size # the model selects (classify) one of 256 classes (ASCII codes) per time unit
learning_rate=0.1
buckets=[(12, 12)] # Because seq2seq does batch learning, it buckets input by length. This time we will only deal with one bucket.
PAD=[0] # If the input / target sentence is smaller than the bucket size, pad 0.
GO=[1] # Decoder RNN puts the symbol GO as the first input.
batch_size=1
input_string = "Hello World" 
target_string = "How are you"

input_PAD_size = buckets[0][0] - len(input_string) # Decide how much PAD you want to input/
target_PAD_size = buckets[0][0] - len(target_string) - 1 # Decide how much PAD you want to target.
input_data = (map(ord, input_string) + PAD * input_PAD_size) * batch_size # Change the input text to a list of ASCII codes.
target_data = (GO + map(ord, target_string) + PAD * target_PAD_size) * batch_size # Change target phrase to list of ASCII codes.
target_weights= ([1.0]*12 + [0.0]*0) * batch_size  # The number of actual valid (loss counted) number of characters 
                                                    # excluding PAD in the target sentence.
```

## Set up the architecture
```python
class Seq2Seq(object):
        def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, batch_size):
                self.buckets = buckets
                self.batch_size = batch_size
                self.source_vocab_size = source_vocab_size
                self.target_vocab_size = target_vocab_size

                cell = single_cell = tf.contrib.rnn.GRUCell(size)
                if num_layers > 1:
                 cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)

                # The seq2seq function
                # encoder_inputs: A list of ASCII codes in the input sentence.
                # decoder_inputs: A list of ASCII codes in the target sentence.
                # cell: RNN cell to use for seq2seq.
                # num_encoder_symbols, num_decoder_symbols: The number of symbols in the input sentence and the target sentence.
                # embedding_size: Size to embed each ASCII code.
                # feed_previous: Inference (true for learning / false for Inference)
                def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                        return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                                        encoder_inputs, decoder_inputs, cell,
                                        num_encoder_symbols=source_vocab_size,
                                        num_decoder_symbols=target_vocab_size,
                                        embedding_size=size,
                                        feed_previous=do_decode)
                    
                # computational graph 
                self.encoder_inputs = []
                self.decoder_inputs = []
                self.target_weights = []
                
                # Bucket size + one as decoder input node. (One additional creation is because the target symbol is equivalent to the decoder input shifting one space)
                for i in xrange(buckets[-1][0]):
                    self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='encoder{0}'.format(i)))

                for i in xrange(buckets[-1][1] + 1):
                    self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='decoder{0}'.format(i)))
                    self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name='weights{0}'.format(i)))
                    
                # The target symbol is equivalent to the decoder input shifted by one space.
                targets = [self.decoder_inputs[i+1] for i in xrange(len(self.decoder_inputs) - 1)]
                
                # Using seq2seq with buckets
                self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                self.encoder_inputs, self.decoder_inputs, targets,
                                self.target_weights, buckets,
                                lambda x, y: seq2seq_f(x, y, False))
                                
                # Gradient
                self.updates = []
                self.updates.append(tf.train.AdamOptimizer(learning_rate).minimize(self.losses[0]))

        def step(self, session, encoder_inputs, decoder_inputs, target_weights, test):
                bucket_id=0 # Choosing bukcet to use
                encoder_size, decoder_size = self.buckets[bucket_id]

                # Input feed: encoder inputs, decoder inputs, target_weights 
                input_feed = {}
                for l in xrange(encoder_size):
                    input_feed[self.encoder_inputs[l].name] = [encoder_inputs[l]]
                for l in xrange(decoder_size):
                    input_feed[self.decoder_inputs[l].name] = [decoder_inputs[l]]
                    input_feed[self.target_weights[l].name] = [target_weights[l]]

                # Insert a value because there is one more decoder input node created.
                last_target = self.decoder_inputs[decoder_size].name
                input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
                last_weight = self.target_weights[decoder_size].name
                input_feed[last_weight] = np.zeros([self.batch_size], dtype=np.int32)
                

                if not test:
                        output_feed = [self.updates[bucket_id], self.losses[bucket_id]]
                else:
                        output_feed = [self.losses[bucket_id]]  # Loss for this batch.
                        for l in xrange(decoder_size):  # Output logits.
                                output_feed.append(self.outputs[bucket_id][l])


                outputs = session.run(output_feed, input_feed)
                if not test:
                        return outputs[0], outputs[1] # loss
                else:
                        return outputs[0], outputs[1:] # loss, outputs.
```
                        
## Run the model
```python
step=0
test_step=1
with tf.Session(config=config) as session:
        model= Seq2Seq(vocab_size, target_vocab_size, buckets, size=5, num_layers=1, batch_size=batch_size)
        session.run(tf.global_variables_initializer())
        while True:
                model.step(session, input_data, target_data, target_weights, test=False) # no outputs in training
                if step % test_step == 0:
                        test()
                step=step+1
```
                
## Results
```python
step 0, losses 3.343728, output: Hello World -> //          ?
step 1, losses 4.671622, output: Hello World -> ooo?
step 2, losses 7.151422, output: Hello World -> Hooooooooooo?
step 3, losses 6.154211, output: Hello World -> Hww         ?
step 4, losses 4.290471, output: Hello World -> HHHwwwaaawww?
step 5, losses 3.866860, output: Hello World -> wrryyyyyyyyy?
step 6, losses 2.886174, output: Hello World -> HHHrrrrrrHHH?
step 7, losses 2.809648, output: Hello World -> oouuuuuuuouu?
step 8, losses 2.535098, output: Hello World -> wwwwwwwwewww?
step 9, losses 1.986781, output: Hello World -> oooooeeoeooo?
step 10, losses 1.883625, output: Hello World ->     eeeeeeee?
step 11, losses 1.460474, output: Hello World -> HH  ayyoo?
step 12, losses 1.545925, output: Hello World -> HH   yyoooo?
step 13, losses 1.251442, output: Hello World -> H a ayyoyoy?
step 14, losses 1.133996, output: Hello World -> ooa aeeor?
step 15, losses 0.782789, output: Hello World -> oow areoro?
step 16, losses 0.862219, output: Hello World -> HHwwwrewrou?
step 17, losses 0.496292, output: Hello World -> How are ro?
step 18, losses 0.445926, output: Hello World -> oo  aee yo?
step 19, losses 0.359356, output: Hello World -> Hoa are you?
step 20, losses 0.276127, output: Hello World -> How are you?
>>>>> success! Hello World -> How are you? <<<<<<<
```
