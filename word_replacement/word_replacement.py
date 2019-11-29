###
# Copied and modified from bert-japanese/src/extract_features.py
# Replaces one token in a sentence with [MASK] and embeds the predicted word to the original sentence
###

# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
import re
import numpy as np
from numpy import int32, uint32
import itertools

SOURCE_PATH=Path(__file__).absolute()

import sys
PARENT_DIR=SOURCE_PATH.parent.parent
BERT_JA_DIR=PARENT_DIR/"bert-japanese"
sys.path.append(str(BERT_JA_DIR/"bert"))
sys.path.append(str(BERT_JA_DIR/"src"))

import codecs
import collections
import json
import re
import pickle

import modeling
import tokenization as tokenizationOrig
import tokenization_sentencepiece as tokenizationSp
import tensorflow as tf


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_float(
    "mask_prob", 0.05,
    "Probability of a token being masked.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_integer(
    "do_lower_case", 1,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")

# Adeed to use the sentencepiece model
flags.DEFINE_string("model_file", None,
                    "The model file that the SentencePiece model was trained on.")

flags.DEFINE_integer("iter_size", 1000, "Iteration size.")
flags.DEFINE_string("output_epoch_dir", None, "")
flags.DEFINE_string("rand_state_file", None, "Save random state for reproducibility")
flags.DEFINE_string("init_rand_state_file", None, "Random state is initialized to this values")
flags.DEFINE_integer("save_output_iters", 0, "How often to save the output. 0 (default) for not saving.")
flags.DEFINE_string("model_type", None, "Model type. Only for my script.")


class InputExample(object):

  def __init__(self, unique_id, text_a):
    self.unique_id = unique_id
    self.text_a = text_a
    self.reset()

  def reset(self):
    self.tokens_a=None
    self.tokens_b=None

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, masked_lm_positions):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids
    self.masked_lm_positions = masked_lm_positions


def input_fn_builder(features, seq_length, max_predictions_per_seq):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []
  all_masked_lm_positions = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)
    masked_lm_positions=feature.masked_lm_positions
    while len(masked_lm_positions)<max_predictions_per_seq: masked_lm_positions.append(0)
    all_masked_lm_positions.append(masked_lm_positions)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "masked_lm_positions":
            tf.constant(
                all_masked_lm_positions,
                shape=[num_examples, max_predictions_per_seq],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, use_tpu,
                     use_one_hot_embeddings, batch_size):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]
    masked_lm_positions = features["masked_lm_positions"]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))

    logits= get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions)
    numMask=masked_lm_positions.shape[1]
    logits=tf.reshape(logits, (batch_size, numMask, logits.shape[-1]))
    
    tvars = tf.trainable_variables()
    scaffold_fn = None
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)
    if use_tpu:

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

#     all_layers = model.get_all_encoder_layers()

    predictions = {
        "unique_id": unique_ids,
        "logits": logits,
    }

#     for (i, layer_index) in enumerate(layer_indexes):
#       predictions["layer_output_%d" % i] = all_layers[layer_index]

    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer, modelSpecific):
  """Loads a data file into a list of `InputBatch`s."""
  tokenization=modelSpecific.tokenization
  
  features = []
  for (ex_index, example) in enumerate(examples):
#     tokens_a = tokenizer.tokenize(example.text_a)
    tokens_a = example.tokens_a

    tokens_b = None
#     if example.text_b:
#       tokens_b = tokenizer.tokenize(example.text_b)
    if example.tokens_b:
      tokens_b = example.tokens_b

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        input_type_ids.append(1)
      tokens.append("[SEP]")
      input_type_ids.append(1)

    masked_lm_positions = [i for i,tok in enumerate(tokens) if tok=="[MASK]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    if ex_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s" % (example.unique_id))
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

    features.append(
        InputFeatures(
            unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
            masked_lm_positions=masked_lm_positions,))
  return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def read_examples(input_file, modelSpecific):
  """Read a list of `InputExample`s from an input file."""
  tokenization=modelSpecific.tokenization
  examples = []
  unique_id = 0
  with tf.gfile.GFile(input_file, "r") as reader:
    while True:

      line = tokenization.convert_to_unicode(reader.readline())
      if not line:
        break
    
      line = line.strip()
      text_a = None
      text_b = None
      m = re.match(r"^(.*) \|\|\| (.*)$", line)

      if m is None:
        text_a = line
      else:
        text_a = m.group(1)
        text_b = m.group(2)

      examples.append(
#           InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
          InputExample(unique_id=unique_id, text_a=text_a)) #if text_b exists, it is added as a new sentence.
      unique_id += 1
      
      if text_b is not None:
        examples.append(InputExample(unique_id=unique_id, text_a=text_b))
        unique_id += 1


  return examples


def main(_):
  modelSpecific=ModelSpecificConfig()
  
  if FLAGS.init_rand_state_file is not None:
    with open(FLAGS.init_rand_state_file, "rb") as f: np.random.set_state(pickle.load(f))
  if FLAGS.rand_state_file is not None:
    with open(FLAGS.rand_state_file, "wb") as f: pickle.dump(np.random.get_state(), f)
  
#   layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

#   bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  bert_config = modelSpecific.loadBertConfig()

#  tokenizer = tokenization.FullTokenizer(
#      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
#   tokenizer = tokenization.FullTokenizer(
#       model_file=FLAGS.model_file, vocab_file=FLAGS.vocab_file,
#       do_lower_case=FLAGS.do_lower_case)
  tokenization=modelSpecific.tokenization
  tokenizer = modelSpecific.tokenizer

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      master=FLAGS.master,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  examples = read_examples(FLAGS.input_file, modelSpecific)
  
  batch_size=FLAGS.batch_size
  if len(examples)<batch_size: batch_size=len(examples)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings, batch_size=batch_size)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=batch_size)
  
  vocab=tokenization.load_vocab(FLAGS.vocab_file)
  inv_vocab = {v: k for k, v in vocab.items()}
  
  for it in range(FLAGS.iter_size):
    print("Iteration", it)
    singleIter(examples, tokenizer, estimator, inv_vocab, modelSpecific)
    generatedTexts=[example.text_a for example in examples]
    if FLAGS.save_output_iters>0 and it%FLAGS.save_output_iters==0:
      saveGenerated(generatedTexts, FLAGS.output_dir+"/Iter"+str(it)+".txt")
    examples=examples
    
  saveGenerated(generatedTexts, FLAGS.output_file)


def singleIter(examples, tokenizer, estimator, inv_vocab, modelSpecific):
  for ei,example in enumerate(examples):
    example.reset()
    
  for ei,example in enumerate(examples):
    tokens = tokenizer.tokenize(example.text_a)
    example.tokens=list(tokens)
    example.tokensMask=list(tokens)
    modelSpecific.makeTokens0(example)
    
    numMask=int(np.ceil(len(example.tokens)*FLAGS.mask_prob))
    numMask=min(numMask, FLAGS.max_predictions_per_seq)
    example.maskIndex=np.sort(np.random.choice(len(tokens), numMask, replace=False))
    for mi in example.maskIndex: example.tokensMask[mi]="[MASK]"
  
  for ei,example in enumerate(examples):
    appendedIndex=[ei,]
    length=len(example.tokens)+3 #+3 for [CLS]+[SEP]*2
    while True:
      nextIndex=[]
      ni=min(appendedIndex)-1
      if ni>=0 and length+len(examples[ni].tokens)<=FLAGS.max_seq_length: nextIndex.append(ni)
      ni=max(appendedIndex)+1
      if ni<len(examples) and length+len(examples[ni].tokens)<=FLAGS.max_seq_length: nextIndex.append(ni)
      if len(nextIndex)==0: break
      if len(nextIndex)==1: nextIndex=nextIndex[0]
      else: nextIndex=np.random.choice(nextIndex)
      appendedIndex.append(nextIndex)
      length+=len(examples[nextIndex].tokens)
    
    appendedIndex=sorted(appendedIndex)
    sepBeforeSentence=bool(np.random.randint(2))
    thisIndex=appendedIndex.index(ei)
    if sepBeforeSentence: sizeA=thisIndex
    else: sizeA=thisIndex+1
    tokens0=example.tokens
    example.tokens=example.tokensMask
    if sizeA>0:
      tokens_a=list(itertools.chain.from_iterable([examples[ai].tokens for ai in appendedIndex[:sizeA]]))
    sizeB=len(appendedIndex)-sizeA
    if sizeB>0:
      tokens_b=list(itertools.chain.from_iterable([examples[ai].tokens for ai in appendedIndex[sizeA:]]))
    example.tokens=tokens0
    if sizeA>0:
      example.tokens_a=tokens_a
      if sizeB>0:
        example.tokens_b=tokens_b
    else:
      assert sizeB>0
      example.tokens_a=tokens_b
  
  features = convert_examples_to_features(
      examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer, modelSpecific=modelSpecific)

  unique_id_to_example = {}
  for example in examples:
    unique_id_to_example[example.unique_id] = example

  max_predictions_per_seq=max([len(example.maskIndex) for example in examples])
  input_fn = input_fn_builder(
      features=features, seq_length=FLAGS.max_seq_length, max_predictions_per_seq=max_predictions_per_seq)

  for result in estimator.predict(input_fn, yield_single_examples=True):
    unique_id = int(result["unique_id"])
    example = unique_id_to_example[unique_id]
    logits = result["logits"]
    for mii,mi in enumerate(example.maskIndex):
      word0=example.tokens[mi]
      for index in (-logits[mii]).argsort():
        word=inv_vocab[index]
#         print(example.unique_id, mi, word0, word)
        if word!=modelSpecific.unk_token and word!=word0: break
      modelSpecific.replaceWord(example, mi, word)
    text=modelSpecific.joinTokens(example)
    if not modelSpecific.checkPunc(text): text+=modelSpecific.defaultSentencePunctuation
    example.text_a=text


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions):
  # Copied from run_pretraining.py
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
  return logits


def gather_indexes(sequence_tensor, positions):
  # Copied from run_pretraining.py
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


class ModelSpecificConfig:
  MODEL_TYPES=set(("ja-sp", "orig"))
  
  def __init__(self):
    assert FLAGS.model_type in ModelSpecificConfig.MODEL_TYPES
    
    if FLAGS.model_type=="ja-sp":
      self._init_ja_sp()
      self.loadBertConfig=self._loadBertConfig_ja_sp
      self.makeTokens0=self._makeTokens0_ja_sp
      self.checkPunc=self._checkPunc_ja_sp
      self.replaceWord=self._replaceWord_ja_sp
      self.joinTokens=self._joinTokens_ja_sp
    elif FLAGS.model_type=="orig":
      self._init_orig()
      self.loadBertConfig=self._loadBertConfig_orig
      self.checkPunc=self._checkPunc_orig
      self.makeTokens0=self._makeTokens0_orig
      self.replaceWord=self._replaceWord_orig
      self.joinTokens=self._joinTokens_orig

  def _init_ja_sp(self):
    fileSentencePunctuation=Path(__file__).parent/"SentencePunctuation-ja.txt"
    self.sentencePunctuation,self.defaultSentencePunctuation=self._readSentencePunctuation(fileSentencePunctuation)
    
    self.tokenization=tokenizationSp
    self.tokenizer = self.tokenization.FullTokenizer(
      model_file=FLAGS.model_file, vocab_file=FLAGS.vocab_file,
      do_lower_case=FLAGS.do_lower_case)
    
    self.unk_token="<unk>"

  def _init_orig(self):
    FLAGS.do_lower_case=0
    
    self.defaultSentencePunctuation="."
    
    self.tokenization=tokenizationOrig
    self.tokenizer = self.tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    
    self.unk_token="[UNK]" #see bert/tokenization/WordpieceTokenizer

  def _loadBertConfig_ja_sp(self):
    from utils import str_to_value
    import configparser
    CONFIGPATH = str(BERT_JA_DIR/"config.ini")
    config = configparser.ConfigParser()
    config.read(CONFIGPATH)
  
    import tempfile
    bert_config_file = tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8', suffix='.json')
    bert_config_file.write(json.dumps({k:str_to_value(v) for k,v in config['BERT-CONFIG'].items()}))
    bert_config_file.seek(0)
    return modeling.BertConfig.from_json_file(bert_config_file.name)

  def _loadBertConfig_orig(self):
    return modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
   
  def _checkPunc_ja_sp(self, text):
    for sp in self.sentencePunctuation:
      if text.endswith(sp):
        return True
    return False
  
  def _checkPunc_orig(self, text):
    return self.tokenization._is_punctuation(text[-1])
  
  def _readSentencePunctuation(self, fileSentencePunctuation):
    sentencePunctuation=[]
    with open(fileSentencePunctuation, "r", encoding="utf8") as f:
      for line in f:
        line=line.strip()
        if line.startswith("#"): continue
        sentencePunctuation.append(line)
    return set(sentencePunctuation), sentencePunctuation[0]
  
  def _makeTokens0_ja_sp(self, example):
    tokens=example.tokens
    assert len(example.text_a)==sum(map(len, tokens))-1, "".join(map(str,zip(" "+example.text_a, "".join(tokens))))
    example.tokens0=[]
    index=0
    for ti,token in enumerate(example.tokens):
      if ti==0: token=token[1:] #_
      example.tokens0.append(example.text_a[index:index+len(token)])
      index+=len(token)
   
  def _makeTokens0_orig(self, example):
    example.tokens0=[]
    example.tokenIndexToken0Index={}
    index=0
    for ti,token in enumerate(example.tokens):
      if token.startswith("##"): token=token[2:]
      nextIndex=example.text_a[index:].find(token)
      assert nextIndex>=0
      if nextIndex>0:
        example.tokens0.append(example.text_a[index:index+nextIndex])
        index+=nextIndex
      assert example.text_a[index:index+len(token)]==token, example.text_a[index:index+len(token)]+" "+token
      example.tokenIndexToken0Index[ti]=len(example.tokens0)
      example.tokens0.append(token)
      index+=len(token)

  def _replaceWord_ja_sp(self, example, mi, word):
    example.tokens0[mi]=word
    
  def _replaceWord_orig(self, example, mi, word):
    example.tokens0[example.tokenIndexToken0Index[mi]]=word
    
  def _joinTokens_ja_sp(self, example):
    text="".join(example.tokens0)
    text=text.replace("▁", "").strip()
    return text
   
  def _joinTokens_orig(self, example):
    tokens=[]
    for t0i,token0 in enumerate(example.tokens0):
      if token0.startswith("##"):
        while len(tokens)>0 and tokens[-1]==" " and not (len(tokens)>1 and tokenizationOrig._is_punctuation(tokens[-2][-1])): tokens.pop()
        token0=token0[2:]
      tokens.append(token0)
    text="".join(tokens)
    return text


def saveGenerated(generatedTexts, file):
  with open(file, "w", encoding="utf8") as f:
    for text in generatedTexts:
      print(text, file=f)


def runJa():
  tf.logging.set_verbosity(tf.logging.FATAL)
  
  sys.argv.extend(("--model_type", "ja-sp"))
  
  BERT_JA_DIR=PARENT_DIR/"bert-japanese"
  # DIR_BERT_JA_MODEL=BERT_JA_DIR/"model"
  DIR_BERT_JA_MODEL=Path(r"D:\cycentum\BertBasedTextGeneration\bert-japanese\model")
  PRETRAINED_MODEL_PATH = DIR_BERT_JA_MODEL/"model.ckpt-1400000"
  sys.argv.extend(("--init_checkpoint", str(PRETRAINED_MODEL_PATH)))

  VOCAB_FILE=DIR_BERT_JA_MODEL/"wiki-ja.vocab"
  sys.argv.extend(("--vocab_file", str(VOCAB_FILE)))

  SP_MODEL_FILE=DIR_BERT_JA_MODEL/"wiki-ja.model"
  sys.argv.extend(("--model_file", str(SP_MODEL_FILE)))
  ###
  sys.argv.extend(("--input_file", r"D:\cycentum\BertBasedTextGeneration\tmp\InputJ.txt"))
  sys.argv.extend(("--output_file", r"D:\cycentum\BertBasedTextGeneration\tmp\OutputJ.txt"))
  sys.argv.extend(("--random_seed_file", r"D:\cycentum\BertBasedTextGeneration\tmp\SeedJ.txt"))
  sys.argv.extend(("--iter_size", r"1"))
  ###
  
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("vocab_file")
#   flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("output_file")
  
  tf.app.run()
  
  
def runEn():
  tf.logging.set_verbosity(tf.logging.FATAL)
  
  sys.argv.extend(("--model_type", "orig"))
  
  DIR_BERT_MODEL=Path(r"D:\cycentum\BertBasedTextGeneration\bert_model\cased_L-24_H-1024_A-16")
  PRETRAINED_MODEL_PATH = DIR_BERT_MODEL/"bert_model.ckpt"
  sys.argv.extend(("--init_checkpoint", str(PRETRAINED_MODEL_PATH)))

  VOCAB_FILE=DIR_BERT_MODEL/"vocab.txt"
  sys.argv.extend(("--vocab_file", str(VOCAB_FILE)))

  sys.argv.extend(("--bert_config_file", str(DIR_BERT_MODEL/"bert_config.json")))

  ###
  sys.argv.extend(("--input_file", r"D:\cycentum\BertBasedTextGeneration\tmp\InputE.txt"))
  sys.argv.extend(("--output_file", r"D:\cycentum\BertBasedTextGeneration\tmp\OutputE.txt"))
  sys.argv.extend(("--random_seed_file", r"D:\cycentum\BertBasedTextGeneration\tmp\SeedE.txt"))
  sys.argv.extend(("--iter_size", r"1"))
  ###

  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("output_file")
  tf.app.run()



if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.FATAL)
  
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("vocab_file")
#	 flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("output_file")

  tf.app.run(main=main)
