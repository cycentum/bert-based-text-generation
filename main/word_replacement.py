###
# Copied and modified from bert-japanese/src/extract_features.py
# Replaces one token in a sentence with [MASK] and embeds the predicted word to the original sentence
###

# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
SOURCE_PATH=Path(__file__)
import sys
PARENT_DIR=SOURCE_PATH.parent.parent
BERT_JA_DIR=PARENT_DIR/"bert-japanese"
sys.path.append(str(BERT_JA_DIR/"bert"))
sys.path.append(str(BERT_JA_DIR/"src"))
CONFIGPATH = str(BERT_JA_DIR/"config.ini")
PRETRAINED_MODEL_PATH = str(BERT_JA_DIR/"model"/"model.ckpt-1400000")
VOCAB_FILE=str(BERT_JA_DIR/"model"/"wiki-ja.vocab")
SP_MODEL_FILE=str(BERT_JA_DIR/"model"/"wiki-ja.model")

import codecs
import collections
import json
import re
import numpy as np
from numpy import int32, uint32

import modeling
#import tokenization
import tokenization_sentencepiece as tokenization
import tensorflow as tf


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("iter_size", 1000, "Iteration size.")
flags.DEFINE_string("input_file", None, "")
flags.DEFINE_string("output_file", None, "")
flags.DEFINE_string("output_dir", None, "")
flags.DEFINE_string("seed_file", None, "Save random seed for reproducibility")
flags.DEFINE_integer("save_output_iters", 0, "How often to save the output. 0 (default) for not saving.")

import configparser
import glob
import os
import pandas as pd
import subprocess
import sys
import tarfile 
from urllib.request import urlretrieve
import json
import tempfile
import tensorflow as tf


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


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
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



# sys.path.append("../src")
from utils import str_to_value

# sys.path.append("../bert")
import modeling

# CURDIR = os.getcwd()
# CONFIGPATH = os.path.join(CURDIR, os.pardir, 'config.ini')
config = configparser.ConfigParser()
config.read(CONFIGPATH)

import tempfile
bert_config_file = tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8', suffix='.json')
bert_config_file.write(json.dumps({k:str_to_value(v) for k,v in config['BERT-CONFIG'].items()}))
bert_config_file.seek(0)
bert_config_file_path = str(bert_config_file.name)
bert_config = modeling.BertConfig.from_json_file(bert_config_file.name)

flags.DEFINE_string(
#     "bert_config_file", None,
    "bert_config_file", bert_config_file.name,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
#     "init_checkpoint", None,
    "init_checkpoint", PRETRAINED_MODEL_PATH,
    "Initial checkpoint (usually from a pre-trained BERT model).")

# flags.DEFINE_string("vocab_file", None,
flags.DEFINE_string("vocab_file", VOCAB_FILE,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
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
# flags.DEFINE_string("model_file", None,
flags.DEFINE_string("model_file", SP_MODEL_FILE,
                    "The model file that the SentencePiece model was trained on.")

class InputExample(object):

  def __init__(self, unique_id, text):
    self.unique_id = unique_id
    self.text = text


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, masked_lm_positions):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids
    self.masked_lm_positions = masked_lm_positions


def input_fn_builder(features, seq_length):
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
    all_masked_lm_positions.append(feature.masked_lm_positions)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)
    max_predictions_per_seq=1

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
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    masked_lm_positions = features["masked_lm_positions"]
    
#     (masked_lm_loss,
#      masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
    logits= get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
#          masked_lm_positions, masked_lm_ids, masked_lm_weights)
         masked_lm_positions, None, None)

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))

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

    predictions = {
        "unique_id": unique_ids,
        "logits": logits,
    }

#     for (i, layer_index) in enumerate(layer_indexes):
#         predictions["layer_output_%d" % i] = all_layers[layer_index]

    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    thisTokens=list(example.tokens)
    thisTokens[example.maskPosition]="[MASK]"
  	
    tokens=["[CLS]"]
    for eii,ei in enumerate(example.appendedIndex): #assume sorted
    	if ei==ex_index: t=thisTokens
    	else: t=examples[ei].tokens
#     	if eii>0: t=t[1:] #remove _ at t[0]
    	tokens.extend(t)
    	if ei==ex_index: tokens.append("[SEP]")
    if tokens[-1]!="[SEP]": tokens.append("[SEP]")
    assert len(tokens)<=seq_length, str(tokens)
    masked_lm_positions=[tokens.index("[MASK]"),]
    assert "[MASK]" not in tokens[masked_lm_positions[0]+1:]

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
    input_type_ids = []
    type_id=0
    for token in tokens:
      input_type_ids.append(type_id)
      if token=="[SEP]": type_id+=1
    
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
#             input_type_ids=input_type_ids))
            input_type_ids=input_type_ids,
            masked_lm_positions=masked_lm_positions))
#   return features
  return features


def read_examples(input_file):
  """Read a list of `InputExample`s from an input file."""
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
        text_b = m.group(2) # text_b will be ignored, accepting only a single sentence per line

      examples.append(
          InputExample(unique_id=unique_id, text=text_a))
      unique_id += 1

  return examples


def saveGenerated(generatedTexts, file):
  with open(file, "w", encoding="utf8") as f:
    for text in generatedTexts:
    	print(text, file=f)
    	
    	
def readSentencePunctuation():
  sentencePunctuation=[]
  with open(Path(__file__).parent/"SentencePunctuation-ja.txt", "r", encoding="utf8") as f:
  	for line in f:
  		line=line.strip()
  		if line.startswith("#"): continue
  		sentencePunctuation.append(line)
  return set(sentencePunctuation), sentencePunctuation[0]
 

def main(_):
  seed=np.random.randint(np.iinfo(uint32).max, dtype=uint32)
  np.random.seed(seed)
  if FLAGS.seed_file is not None:
	  with open(FLAGS.seed_file, "w") as f: print(seed, file=f)
	
  iterSize= FLAGS.iter_size

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

#  tokenizer = tokenization.FullTokenizer(
#      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tokenizer = tokenization.FullTokenizer(
      model_file=FLAGS.model_file, vocab_file=FLAGS.vocab_file,
      do_lower_case=FLAGS.do_lower_case)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      master=FLAGS.master,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))
  
  examples = read_examples(FLAGS.input_file)
  batch_size=FLAGS.batch_size
  if batch_size<len(examples): batch_size=len(examples)
    
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)
  
  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=batch_size)

  vocab=tokenization.load_vocab(FLAGS.vocab_file)
  inv_vocab = {v: k for k, v in vocab.items()}
  
  sentencePunctuation,defaultSentencePunctuation=readSentencePunctuation()
  	
  for it in range(iterSize):
    print("Iteration", it)
    
    maskIndexAll=np.empty((len(examples)), int32)
    maskedWordAll=[]
    for ei,example in enumerate(examples):
    	tokens = tokenizer.tokenize(example.text)
    	if len(tokens[0])>1: #force first token to be _
    	  tokens.insert(0, tokens[0][:1])
    	  tokens[1]=tokens[1][1:]
    	assert len(example.text)==sum(map(len, tokens))-1, "".join(map(str,zip(" "+example.text, "".join(tokens))))
    	example.tokens=tokens
    	
    	maskPosition=np.random.randint(1, len(tokens))
    	maskIndex=sum(map(len,tokens[:maskPosition]))-1 #-1 for _
    	maskIndexAll[ei]=maskIndex
    	example.maskPosition=maskPosition
    	maskedWordAll.append(tokens[maskPosition])
    	
    for ei,example in enumerate(examples):
    	length=len(example.tokens)+3 #+3 for [CLS], [SEP]*2
    	appendedIndex=[ei,]
    	while True: #For taking the sentence-to-sentence relationship into account
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
    	example.appendedIndex=sorted(appendedIndex)
    	
    newWords=singleIter(it, tokenizer, examples, run_config, estimator, inv_vocab, maskedWordAll)
    newTexts=[]
    for ei,example in enumerate(examples):
    	maskIndex=maskIndexAll[ei]
    	word0=maskedWordAll[ei]
    	word=newWords[ei]
    	text0=example.text
    	sentence=text0[:maskIndex]+word+text0[maskIndex+len(word0):]
    	sentence=sentence.strip()
    	if sentence[-1] not in sentencePunctuation: sentence+=defaultSentencePunctuation
    	example.text=sentence
    	newTexts.append(sentence)
    
    if FLAGS.save_output_iters>0 and it%FLAGS.save_output_iters==0:
    	saveGenerated(newTexts, FLAGS.output_dir+"/Iter"+str(it)+".txt")
  
  saveGenerated(newTexts, FLAGS.output_file)


def singleIter(it, tokenizer, examples, run_config, estimator, inv_vocab, maskedWordAll):
  features = convert_examples_to_features(
      examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

  unique_id_to_feature_index = {}
  for fi,feature in enumerate(features):
    unique_id_to_feature_index[feature.unique_id] = fi

  input_fn = input_fn_builder(
      features=features, seq_length=FLAGS.max_seq_length)

  predictedWords={}
  for result in estimator.predict(input_fn, yield_single_examples=True):
    unique_id = int(result["unique_id"])
    featureIndex=unique_id_to_feature_index[unique_id]
    maskedWord=maskedWordAll[featureIndex]
    logits=result["logits"]
    for index in (-logits).argsort(): #most likely word other than <unk>, _, or the original word
    	word=inv_vocab[index]
    	word=word.replace("▁", "")
    	word="".join(word.split()) #remove white spaces
    	if len(word)>0 and word!="<unk>" and word!=maskedWord: break
    predictedWords[featureIndex]=word
  return predictedWords

if __name__ == "__main__":
	flags.mark_flag_as_required("input_file")
	flags.mark_flag_as_required("vocab_file")
	flags.mark_flag_as_required("bert_config_file")
	flags.mark_flag_as_required("init_checkpoint")
	flags.mark_flag_as_required("output_file")
	flags.mark_flag_as_required("iter_size")
	tf.app.run()
