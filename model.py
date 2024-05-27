from http.server import HTTPServer
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import unicodedata
import re
import os
import io
import time
from custom_data import load_custom_data
import nltk.translate.bleu_score as bleu
import numpy as np


class NMTDataset:
  def __init__(self):
      self.inp_lang_tokenizer = None
      self.targ_lang_tokenizer = None

  def unicode_to_ascii(self, s):
      return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

  def preprocess_sentence(self, sequence):
      sequence = self.unicode_to_ascii(sequence.lower().strip())
      sequence = re.sub(r"([?.!,¿])", r" \1 ", sequence)
      sequence = re.sub(r'[" "]+', " ", sequence)
      sequence = sequence.strip()
      sequence = '<start> ' + sequence + ' <end>'
      return sequence

  def create_dataset(self, num_examples):
      data = load_custom_data()
      word_pairs = [[self.preprocess_sentence(x["query"]), self.preprocess_sentence(x["question"])] for x in data[:num_examples]]
      return zip(*word_pairs)

  def tokenize(self, lang):
      lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
      lang_tokenizer.fit_on_texts(lang)

      tensor = lang_tokenizer.texts_to_sequences(lang) 
      tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

      return tensor, lang_tokenizer

  def load_dataset(self, num_examples=None):
      targ_lang, inp_lang = self.create_dataset(num_examples)
      input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
      target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)

      return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

  def call(self, num_examples, BUFFER_SIZE, BATCH_SIZE):
      input_tensor, target_tensor, self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.load_dataset(num_examples)

      input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

      train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
      train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

      val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
      val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

      return train_dataset, val_dataset, self.inp_lang_tokenizer, self.targ_lang_tokenizer
  
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
      super(Encoder, self).__init__()
      self.batch_sz = batch_sz
      self.enc_units = enc_units
      self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

      ##-------- LSTM layer in Encoder ------- ##
      self.lstm_layer = tf.keras.layers.LSTM(self.enc_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
      x = self.embedding(x)
      output, h, c = self.lstm_layer(x, initial_state = hidden)
      return output, h, c

    def initialize_hidden_state(self):
      return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, max_length_input, max_length_output, attention_type='luong'):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.attention_type = attention_type
    self.max_length_input=max_length_input
    self.max_length_output=max_length_output

    # Embedding Layer
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    #Final Dense layer on which softmax will be applied
    self.fc = tf.keras.layers.Dense(vocab_size)

    # Define the fundamental cell for decoder recurrent structure
    self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)


    # Sampler
    self.sampler = tfa.seq2seq.sampler.TrainingSampler()

    # Create attention mechanism with memory = None
    self.attention_mechanism = self.build_attention_mechanism(self.dec_units, 
                                                              None, self.batch_sz*[max_length_input], self.attention_type)

    # Wrap attention mechanism with the fundamental rnn cell of decoder
    self.rnn_cell = self.build_rnn_cell(batch_sz)

    # Define the decoder with respect to fundamental rnn cell
    self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)


  def build_rnn_cell(self, batch_sz):
    rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, 
                                  self.attention_mechanism, attention_layer_size=self.dec_units)
    return rnn_cell

  def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
    if(attention_type=='bahdanau'):
      return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
    else:
      return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

  def build_initial_state(self, batch_sz, encoder_state, Dtype):
    decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
    return decoder_initial_state


  def call(self, inputs, initial_state):
    x = self.embedding(inputs)
    outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[self.max_length_output-1])
    return outputs
    


class Seq2seq():

  BUFFER_SIZE = 32000
  BATCH_SIZE = 64
  num_examples = 1000

  dataset_creator = NMTDataset()
  train_dataset, val_dataset, inp_lang, targ_lang = dataset_creator.call(num_examples, BUFFER_SIZE, BATCH_SIZE)

  example_input_batch, example_target_batch = next(iter(train_dataset))

  vocab_inp_size = len(inp_lang.word_index)+1
  vocab_tar_size = len(targ_lang.word_index)+1
  max_length_input = example_input_batch.shape[1]
  max_length_output = example_target_batch.shape[1]

  embedding_dim = 256
  units = 1024
  steps_per_epoch = num_examples//BATCH_SIZE

  ## Test Encoder Stack
  encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
  sample_hidden = encoder.initialize_hidden_state()
  sample_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)


  # Test decoder stack
  decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, max_length_input, max_length_output, 'luong')
  sample_x = tf.random.uniform((BATCH_SIZE, max_length_output))
  decoder.attention_mechanism.setup_memory(sample_output)
  initial_state = decoder.build_initial_state(BATCH_SIZE, [sample_h, sample_c], tf.float32)
  sample_decoder_outputs = decoder(sample_x, initial_state)
  optimizer = tf.keras.optimizers.Adam()

  def loss_function(real, pred):
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = cross_entropy(y_true=real, y_pred=pred)
    mask = tf.logical_not(tf.math.equal(real,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)  
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss



  checkpoint_dir = './saved_model'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                  encoder=encoder,
                                  decoder=decoder)



  @tf.function
  def train_step(self, inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
      enc_output, enc_h, enc_c = self.encoder(inp, enc_hidden)


      dec_input = targ[ : , :-1 ]     # Ignore <end> token
      real = targ[ : , 1: ]           # ignore <start> token

      # Set the AttentionMechanism object with encoder_outputs
      self.decoder.attention_mechanism.setup_memory(enc_output)

      # Create AttentionWrapperState as initial_state for decoder
      decoder_initial_state = self.decoder.build_initial_state(self.BATCH_SIZE, [enc_h, enc_c], tf.float32)
      pred = self.decoder(dec_input, decoder_initial_state)
      logits = pred.rnn_output
      loss = self.loss_function(real, logits)

    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))

    return loss


  def train(self):
    EPOCHS = 20
    for epoch in range(EPOCHS):
      start = time.time()

      enc_hidden = self.encoder.initialize_hidden_state()
      total_loss = 0
      # print(enc_hidden[0].shape, enc_hidden[1].shape)

      for (batch, (inp, targ)) in enumerate(self.train_dataset.take(self.steps_per_epoch)):
        batch_loss = self.train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
      print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / self.steps_per_epoch))
      print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
    self.checkpoint.save(file_prefix = self.checkpoint_prefix)


  def tokens_to_sentence(self, tokens, tokenizer):
    sentence = ' '.join([tokenizer.index_word[token] for token in tokens if token != 0])
    return sentence
  
  def evaluate_sentence(self, sentence):
    sentence = self.dataset_creator.preprocess_sentence(sentence)

    inputs = [self.inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                            maxlen=self.max_length_input,
                                                            padding='post')
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]
    result = ''

    enc_start_state = [tf.zeros((inference_batch_size, self.units)), tf.zeros((inference_batch_size, self.units))]
    enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

  #   dec_h = enc_h
  #   dec_c = enc_c

    start_tokens = tf.fill([inference_batch_size], self.targ_lang.word_index['<start>'])
    end_token = self.targ_lang.word_index['<end>']

    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    # Instantiate BasicDecoder object
    decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.rnn_cell, sampler=greedy_sampler, output_layer=self.decoder.fc)
    # Setup Memory in decoder stack
    self.decoder.attention_mechanism.setup_memory(enc_out)

    # set decoder_initial_state
    decoder_initial_state = self.decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)


    ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder 
    ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this. 
    ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

    decoder_embedding_matrix = self.decoder.embedding.variables[0]

    outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token= end_token, initial_state=decoder_initial_state)
    return outputs.sample_id.numpy()

  def translate(self, sentence):
    result = self.evaluate_sentence(sentence)
    result = self.targ_lang.sequences_to_texts(result)
    return result


  def beam_evaluate_sentence(self, sentence, beam_width=3):
    sentence = self.dataset_creator.preprocess_sentence(sentence)

    inputs = [self.inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                            maxlen=self.max_length_input,
                                                            padding='post')
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]
    result = ''

    enc_start_state = [tf.zeros((inference_batch_size, self.units)), tf.zeros((inference_batch_size,self.units))]
    enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

    dec_h = enc_h
    dec_c = enc_c

    start_tokens = tf.fill([inference_batch_size], self.targ_lang.word_index['<start>'])
    end_token = self.targ_lang.word_index['<end>']

    # From official documentation
    # NOTE If you are using the BeamSearchDecoder with a cell wrapped in AttentionWrapper, then you must ensure that:
    # The encoder output has been tiled to beam_width via tfa.seq2seq.tile_batch (NOT tf.tile).
    # The batch_size argument passed to the get_initial_state method of this wrapper is equal to true_batch_size * beam_width.
    # The initial state created with get_initial_state above contains a cell_state value containing properly tiled final state from the encoder.

    enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
    self.decoder.attention_mechanism.setup_memory(enc_out)
    print("beam_with * [batch_size, max_length_input, rnn_units] :  3 * [1, 16, 1024]] :", enc_out.shape)

    # set decoder_inital_state which is an AttentionWrapperState considering beam_width
    hidden_state = tfa.seq2seq.tile_batch([enc_h, enc_c], multiplier=beam_width)
    decoder_initial_state = self.decoder.rnn_cell.get_initial_state(batch_size=beam_width*inference_batch_size, dtype=tf.float32)
    decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)

    # Instantiate BeamSearchDecoder
    decoder_instance = tfa.seq2seq.BeamSearchDecoder(self.decoder.rnn_cell,beam_width=beam_width, output_layer=self.decoder.fc)
    decoder_embedding_matrix = self.decoder.embedding.variables[0]

    # The BeamSearchDecoder object's call() function takes care of everything.
    outputs, final_state, sequence_lengths = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
    final_outputs = tf.transpose(outputs.predicted_ids, perm=(0,2,1))
    beam_scores = tf.transpose(outputs.beam_search_decoder_output.scores, perm=(0,2,1))

    return final_outputs.numpy(), beam_scores.numpy()


  # train()

  def loadModel(self):
    status = self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
    status.expect_partial()


  def evaluate_model(self):
    def generate_translations(model, dataset):
        translations = []
        references = []
        for input_batch, target_batch in dataset:
      
            # Generate translations for each sentence in the batch
            for i in range(input_batch.shape[0]):
                input_tensor = input_batch[i].numpy()
                reference_tensor = target_batch[i].numpy()
                
                sentence = self.tokens_to_sentence(input_tensor, model.inp_lang)
                ref_query = self.tokens_to_sentence(reference_tensor, model.targ_lang)
                translation = model.translate(sentence)
                
                translations.append(translation[0].split())
                references.append(ref_query.split()[1:-1])  # Remove <start> and <end> tokens

        return translations, references

    #Generate translations using your model on the validation dataset
    model_translations, reference_translations = generate_translations(self, self.val_dataset)
    bleu_scores = []
    for ref, hyp in zip(reference_translations, model_translations):
        bleu_score = bleu.sentence_bleu([ref], hyp)
        bleu_scores.append(bleu_score)

    average_bleu_score = np.mean(bleu_scores)
    print("Average BLEU Score on Validation Dataset:", average_bleu_score)