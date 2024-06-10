import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import unicodedata
import re
import os
import time
from train_data import load_custom_data
import nltk.translate.bleu_score as bleu
import numpy as np


class NMTDataset:
    def __init__(self):
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None

    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self, word):
        word = self.unicode_to_ascii(word.lower().strip())
        word = re.sub(r"([?.!,¿])", r" \1 ", word)
        word = re.sub(r'[" "]+', " ", word)
        word = word.strip()
        word = '<start> ' + word + ' <end>'
        return word

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
    self.max_length_outputttt = max_length_output
    a = max_length_output
    self.max_length_input = max_length_input

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)


    self.sampler = tfa.seq2seq.sampler.TrainingSampler()
    self.attention_mechanism = self.build_attention_mechanism(self.dec_units, 
                                                              None, self.batch_sz*[max_length_input], self.attention_type)

    self.rnn_cell = self.build_rnn_cell(batch_sz)
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
    outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[self.max_length_outputttt-1])
    return outputs
  

class Seq2Seq():

  def __init__(self):
    self.BUFFER_SIZE = 32000
    self.BATCH_SIZE = 64
    self.num_examples = 1000

    self.dataset_creator = NMTDataset()
    self.train_dataset, self.val_dataset, self.inp_lang, self.targ_lang = self.dataset_creator.call(self.num_examples, self.BUFFER_SIZE, self.BATCH_SIZE)
    self.example_input_batch, self.example_target_batch = next(iter(self.train_dataset))

    self.vocab_inp_size = len(self.inp_lang.word_index)+1
    self.vocab_tar_size = len(self.targ_lang.word_index)+1
    self.max_length_input = self.example_input_batch.shape[1]
    self.max_length_output = self.example_target_batch.shape[1]
    self.embedding_dim = 256
    self.units = 1024
    self.steps_per_epoch = self.num_examples//self.BATCH_SIZE

    self.encoder = Encoder(self.vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)
    self.sample_hidden = self.encoder.initialize_hidden_state()
    self.sample_output, self.sample_h, self.sample_c = self.encoder(self.example_input_batch, self.sample_hidden)

    self.decoder = Decoder(self.vocab_tar_size, self.embedding_dim, self.units, self.BATCH_SIZE, self.max_length_input, self.max_length_output, 'luong')
    self.sample_x = tf.random.uniform((self.BATCH_SIZE, self.max_length_output))
    self.decoder.attention_mechanism.setup_memory(self.sample_output)
    self.initial_state = self.decoder.build_initial_state(self.BATCH_SIZE, [self.sample_h, self.sample_c], tf.float32)
    self.sample_decoder_outputs = self.decoder(self.sample_x, self.initial_state)
    self.optimizer = tf.keras.optimizers.Adam()

    self.checkpoint_dir = './saved_model'
    self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
    self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                  encoder=self.encoder,
                                  decoder=self.decoder)

    #END OF __INIT__

  def loss_function(self, real, pred):
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = cross_entropy(y_true=real, y_pred=pred)
    mask = tf.logical_not(tf.math.equal(real,0)) 
    mask = tf.cast(mask, dtype=loss.dtype)  
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss


  @tf.function
  def train_step(self, inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
      enc_output, enc_h, enc_c = self.encoder(inp, enc_hidden)

      dec_input = targ[ : , :-1 ]     # Ignore <end> token
      real = targ[ : , 1: ]           # ignore <start> token

      self.decoder.attention_mechanism.setup_memory(enc_output)
      decoder_initial_state = self.decoder.build_initial_state(self.BATCH_SIZE, [enc_h, enc_c], tf.float32)
      pred = self.decoder(dec_input, decoder_initial_state)
      logits = pred.rnn_output
      loss = self.loss_function(real, logits)

    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))

    return loss


  def train(self):
      EPOCHS = 30
      for epoch in range(EPOCHS):
          start = time.time()

          enc_hidden = self.encoder.initialize_hidden_state()
          total_loss = 0

          for (batch, (inp, targ)) in enumerate(self.train_dataset.take(self.steps_per_epoch)):
              batch_loss = self.train_step(inp, targ, enc_hidden)
              total_loss += batch_loss

              if batch % 100 == 0:
                  print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                          batch,
                                                          batch_loss.numpy()))
          print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                              total_loss / self.steps_per_epoch))
          print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
      
      self.checkpoint.save(file_prefix = self.checkpoint_prefix)



  def evaluate_sentence(self, sentence):
    sentence = self.dataset_creator.preprocess_sentence(sentence)

    oov_token = self.inp_lang.word_index['<OOV>']
    inputs = [self.inp_lang.word_index.get(word, oov_token) for word in sentence.split(' ')]
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
    self.decoder.attention_mechanism.setup_memory(enc_out)
    decoder_initial_state = self.decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)

    decoder_embedding_matrix = self.decoder.embedding.variables[0]

    outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token= end_token, initial_state=decoder_initial_state)
    return outputs.sample_id.numpy()

  def translate(self, sentence):
    result = self.evaluate_sentence(sentence)
    result = self.targ_lang.sequences_to_texts(result)
    return result


  def unicode_to_ascii(self, s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

  def pre_process(self, question):
    question = self.unicode_to_ascii(question.lower().strip())
    question = re.sub(r"([?.!,¿])", r" \1 ", question)
    question = re.sub(r'[" "]+', " ", question)
    question = question.strip()

    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    months_replaced = []
    processed_question = []
    for word in question.split(' '):
      if word in months:
        processed_question.append('<month>')
        months_replaced.append(word)
      else:
        processed_question.append(word)

    return ' '.join(processed_question), months_replaced

  def post_process(query, values):
    query = query.removeprefix( '<start> ')
    query = query.removesuffix( ' <end>')
    processed_query = []
    for word in query.split(' '):
      if word == '<month>':
        processed_query.append(values[0])
        values = values[1:]
      else:
        processed_query.append(word)

    processed_query = ' '.join(processed_query)
    print(processed_query)
    pattern = r"' ([^']*) '"
    processed_query = re.sub(pattern, r"'\1'", processed_query)
    return processed_query

  def load_model(self):
    status = self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
    status.expect_partial()


  def evaluate_model(self):
    def generate_translations(model, dataset):
        translations = []
        references = []
        for input_batch, target_batch in dataset:
            for i in range(input_batch.shape[0]):
                input_tensor = input_batch[i].numpy()
                reference_tensor = target_batch[i].numpy()
                
                sentence = self.tokens_to_sentence(input_tensor, model.inp_lang)
                ref_query = self.tokens_to_sentence(reference_tensor, model.targ_lang)
                translation = model.translate(sentence)
                
                translations.append(translation[0].split())
                references.append(ref_query.split()[1:-1])  # Remove <start> and <end> tokens

        return translations, references

    model_translations, reference_translations = generate_translations(self, self.val_dataset)
    bleu_scores = []
    for ref, hyp in zip(reference_translations, model_translations):
        bleu_score = bleu.sentence_bleu([ref], hyp)
        bleu_scores.append(bleu_score)

    average_bleu_score = np.mean(bleu_scores)
    print("Average BLEU Score on Validation Dataset:", average_bleu_score)


# model = Seq2Seq()
# model.train()