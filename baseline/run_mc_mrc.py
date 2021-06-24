# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on RACE and C3."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf
import numpy
import pdb
import sys
from tqdm import tqdm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "json for training. E.g., train.json")

flags.DEFINE_string(
    "predict_file", None,
    "json for predictions. E.g., dev.json or test.json")

flags.DEFINE_string("train_tfrecord", None, "save path for training set tfrecord. For quick load.")
flags.DEFINE_string("predict_tfrecord", None, "save path for prediction set tfrecord. For quick load.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 32,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_string("task_name", "c3", "set task name, shoule be one of `race`, `c3`.")
flags.DEFINE_integer("rand_seed", 12345, "set random seed")
flags.DEFINE_float("loss_lambda", 0.1, "set loss weight for evidence output")

# set random seed (i don't know whether it works or not)
numpy.random.seed(int(FLAGS.rand_seed))
tf.set_random_seed(int(FLAGS.rand_seed))

SPIECE_UNDERLINE = '▁'


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               tokens,
               token_to_orig_map,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               guid=None,
               example_id=None,
               is_real_example=True,
               orig_evidence_text=None,
               evidence_start_position=None,
               evidence_end_position=None):
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.example_id = example_id
    self.guid = guid
    self.is_real_example = is_real_example
    self.orig_evidence_text = orig_evidence_text
    self.evidence_start_position = evidence_start_position
    self.evidence_end_position = evidence_end_position


class InputExample(object):
  """A single training/test example for the RACE dataset."""

  def __init__(self,
               example_id,
               context_sentence,
               doc_tokens,
               start_ending,
               endings,
               label=None,
               orig_evidence_text=None,
               evidence_start_position=None,
               evidence_end_position=None):
    self.example_id = example_id
    self.context_sentence = context_sentence
    self.doc_tokens = doc_tokens
    self.start_ending = start_ending
    self.endings = endings
    self.label = label
    self.orig_evidence_text = orig_evidence_text
    self.evidence_start_position = evidence_start_position
    self.evidence_end_position = evidence_end_position

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    l = [
        "example_id: {}".format(self.example_id),
        "context_sentence: {}".format(self.context_sentence),
        "start_ending: {}".format(self.start_ending),
        "ending_0: {}".format(self.endings[0]),
        "ending_1: {}".format(self.endings[1]),
        "ending_2: {}".format(self.endings[2]),
        "ending_3: {}".format(self.endings[3]),
    ]

    if self.label is not None:
      l.append("label: {}".format(self.label))
      l.append("evidence_start_position: {}".format(self.evidence_start_position))
      l.append("evidence_end_position: {}".format(self.evidence_end_position))

    return ", ".join(l)


class RaceProcessor(object):
  """Processor for the RACE data set."""

  def __init__(self, use_spm, do_lower_case):
    super(RaceProcessor, self).__init__()
    self.use_spm = use_spm
    self.do_lower_case = do_lower_case

  def get_train_examples(self, train_file):
    """Gets a collection of `InputExample`s for the train set."""
    return self.read_examples(train_file, is_training=True)

  def get_dev_examples(self, predict_file):
    """Gets a collection of `InputExample`s for the dev set."""
    return self.read_examples(predict_file, is_training=False)

  def get_labels(self):
    """Gets the list of labels for this data set."""
    return ["A", "B", "C", "D"]

  def process_text(self, text):
    if self.use_spm:
      return tokenization.preprocess_text(text, lower=self.do_lower_case)
    else:
      return tokenization.convert_to_unicode(text)

  def read_examples(self, file_path, is_training=True):
    """Read examples from dream json files."""
    def _is_chinese_char(cp):
      if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
      return False

    def is_fuhao(c):
      if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
                or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
                or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
                or c == '‘' or c == '’':
        return True
      return False

    def _tokenize_chinese_chars(text):
      """Adds whitespace around any CJK character."""
      output = []
      for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or is_fuhao(char):
          if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
            output.append(SPIECE_UNDERLINE)
          output.append(char)
          output.append(SPIECE_UNDERLINE)
        else:
          output.append(char)
      return "".join(output)

    def is_whitespace(c):
      if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
        return True
      return False

    examples = []
    data_file = json.load(tf.gfile.Open(file_path))
    for cur_data in data_file["data"]:
      options = cur_data["options"]
      questions = cur_data["questions"]
      context = self.process_text(cur_data["article"])
      if FLAGS.task_name == "race":
        context_iter = context
      elif FLAGS.task_name == "c3":
        context_iter = _tokenize_chinese_chars(context)
      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = True
      for c in context_iter:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False
        if c != SPIECE_UNDERLINE:
          char_to_word_offset.append(len(doc_tokens) - 1)

      for i in range(len(questions)):
        qa_list = []
        if len(options[i]) != 4:
          for k in range(4-len(options[i])):
            options[i].append("")
        question = self.process_text(questions[i])
        
        for j in range(len(self.get_labels())):
          option = self.process_text(options[i][j])
          if "_" in question:
            qa_cat = question.replace("_", option)
          else:
            qa_cat = " ".join([question, option])
          qa_list.append(qa_cat)

        # evidence
        label = None
        evidence_text = None
        evidence_start_position_final = None
        evidence_end_position_final = None
        if is_training:
          answers = cur_data["answers"]
          evidences = cur_data["evidences"]
          label = ord(answers[i]) - ord("A")
          evidence_text = self.process_text(evidences[i][0])
          if evidence_text == '':
            evidence_start_position_final = 0
            evidence_end_position_final = 0
          else:
            count_i = 0
            repeat_limit = 3
            evidence_start_position = context.find(evidence_text)
            evidence_end_position = evidence_start_position + len(evidence_text) - 1
            while context[evidence_start_position:evidence_end_position + 1] != evidence_text and count_i < repeat_limit:
              evidence_start_position -= 1
              evidence_end_position -= 1
              count_i += 1

            while context[evidence_start_position] == " " or context[evidence_start_position] == "\t" or \
                    context[evidence_start_position] == "\r" or context[evidence_start_position] == "\n":
              evidence_start_position += 1

            evidence_start_position_final = char_to_word_offset[evidence_start_position]
            evidence_end_position_final = char_to_word_offset[evidence_end_position]

            if FLAGS.task_name == "race":
              actual_text = " ".join(doc_tokens[evidence_start_position_final:(evidence_end_position_final + 1)])
              cleaned_evidence_text = " ".join(tokenization.whitespace_tokenize(evidence_text))
            elif FLAGS.task_name == "c3":
              actual_text = "".join(doc_tokens[evidence_start_position_final:(evidence_end_position_final + 1)])
              cleaned_evidence_text = "".join(tokenization.whitespace_tokenize(evidence_text))

            if actual_text.find(cleaned_evidence_text) == -1:
              tf.logging.warning("Could not find evidence: '%s' vs. '%s'", actual_text, cleaned_evidence_text)
              continue

        examples.append(
            InputExample(
                example_id=cur_data["id"] +"-"+str(i),
                context_sentence=context,
                doc_tokens=doc_tokens,
                start_ending=None,
                endings=[qa_list[0], qa_list[1], qa_list[2], qa_list[3]],
                label=label,
                orig_evidence_text=evidence_text,
                evidence_start_position=evidence_start_position_final,
                evidence_end_position=evidence_end_position_final,
            )
        )

    return examples


def convert_single_example(example_index, example, label_size, max_seq_length,
                           tokenizer, max_qa_length, is_training):
  """Loads a data file into a list of `InputBatch`s."""

  # RACE is a multiple choice task. To perform this task using AlBERT,
  # we will use the formatting proposed in "Improving Language
  # Understanding by Generative Pre-Training" and suggested by
  # @jacobdevlin-google in this issue
  # https://github.com/google-research/bert/issues/38.
  #
  # Each choice will correspond to a sample on which we run the
  # inference. For a given RACE example, we will create the 4
  # following inputs:
  # - [CLS] context [SEP] choice_1 [SEP]
  # - [CLS] context [SEP] choice_2 [SEP]
  # - [CLS] context [SEP] choice_3 [SEP]
  # - [CLS] context [SEP] choice_4 [SEP]
  # The model will output a single value for each input. To get the
  # final decision of the model, we will run a softmax over these 4
  # outputs.

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        example_id=-1,
        tokens=[[0] * max_seq_length] * label_size,
        token_to_orig_map= {},
        input_ids=[[0] * max_seq_length] * label_size,
        input_mask=[[0] * max_seq_length] * label_size,
        segment_ids=[[0] * max_seq_length] * label_size,
        label_id=0,
        is_real_example=False)
  else:
    tok_to_orig_index = []
    orig_to_tok_index = []
    token_to_orig_map = {}
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i) 
        all_doc_tokens.append(sub_token)

    for i in range(len(tok_to_orig_index)):
      token_to_orig_map[i] = tok_to_orig_index[i]

    tok_evidence_start_position = None
    tok_evidence_end_position = None
    if is_training:
      tok_evidence_start_position = orig_to_tok_index[example.evidence_start_position]
      if example.evidence_end_position < len(example.doc_tokens) - 1:
        tok_evidence_end_position = orig_to_tok_index[example.evidence_end_position + 1] - 1
      else:
        tok_evidence_end_position = len(all_doc_tokens) - 1
      (tok_evidence_start_position, tok_evidence_end_position) = _improve_answer_span(
          all_doc_tokens, tok_evidence_start_position, tok_evidence_end_position, tokenizer,
          example.orig_evidence_text)

    #context_tokens = tokenizer.tokenize(example.context_sentence)
    context_tokens = all_doc_tokens
    if example.start_ending is not None:
      start_ending_tokens = tokenizer.tokenize(example.start_ending)

    all_input_tokens = []
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    assert len(example.endings) == 4
    for ending in example.endings:
      # We create a copy of the context tokens in order to be
      # able to shrink it according to ending_tokens
      context_tokens_choice = context_tokens[:]
      if example.start_ending is not None:
        ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
      else:
        ending_tokens = tokenizer.tokenize(ending)
      # Modifies `context_tokens_choice` and `ending_tokens` in
      # place so that the total length is less than the
      # specified length.  Account for [CLS], [SEP], [SEP] with
      # "- 3"
      ending_tokens = ending_tokens[- max_qa_length:]

      if len(context_tokens_choice) + len(ending_tokens) > max_seq_length - 3:
        context_tokens_choice = context_tokens_choice[: (
            max_seq_length - 3 - len(ending_tokens))]
        context_tokens_choice_len = len(context_tokens_choice)
      tokens = ["[CLS]"] + context_tokens_choice + (
          ["[SEP]"] + ending_tokens + ["[SEP]"])
      segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (
          len(ending_tokens) + 1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding = [0] * (max_seq_length - len(input_ids))
      input_ids += padding
      input_mask += padding
      segment_ids += padding

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length
      all_input_tokens.append(tokens)
      all_input_ids.append(input_ids)
      all_input_mask.append(input_mask)
      all_segment_ids.append(segment_ids)

    label = example.label
    evidence_start_position = None
    evidence_end_position = None
    if is_training:
      # For training, if our document chunk does not contain an annotation
      # we throw it out, since there is nothing to predict.
      if tok_evidence_start_position == -1 and tok_evidence_end_position == -1:
        evidence_start_position = 0 
        evidence_end_position = 0
      else:
        doc_start = 0
        doc_end = len(context_tokens_choice) - 1
        out_of_span = False
        if not (tok_evidence_start_position >= doc_start and
                tok_evidence_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          evidence_start_position = 0
          evidence_end_position = 0
        else:
          doc_offset = 1 #len(query_tokens) + 2
          evidence_start_position = tok_evidence_start_position - doc_start + doc_offset
          evidence_end_position = tok_evidence_end_position - doc_start + doc_offset

    if example_index < 2:
      tf.logging.info("*** Example ***")
      tf.logging.info("id: {}".format(example.example_id))
      for choice_idx, (tokens, input_ids, input_mask, segment_ids) in \
           enumerate(zip(all_input_tokens, all_input_ids, all_input_mask, all_segment_ids)):
        tf.logging.info("choice: {}".format(choice_idx))
        tf.logging.info("tokens: {}".format(" ".join(tokens)))
        #tf.logging.info("token_to_orig_map: %s" % " ".join(
        #    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
        tf.logging.info(
            "input_ids: {}".format(" ".join(map(str, input_ids))))
        tf.logging.info(
            "input_mask: {}".format(" ".join(map(str, input_mask))))
        tf.logging.info(
            "segment_ids: {}".format(" ".join(map(str, segment_ids))))
        tf.logging.info("label: {}".format(label))

        if is_training:
          evidence_text = " ".join(tokens[evidence_start_position:(evidence_end_position + 1)])
          tf.logging.info("evidence_start_position: %d" % (evidence_start_position))
          tf.logging.info("evidence_end_position: %d" % (evidence_end_position))
          tf.logging.info("evidence: %s" % (tokenization.printable_text(evidence_text)))

    return InputFeatures(
        tokens=all_input_tokens,
        token_to_orig_map=token_to_orig_map,
        example_id=example_index,
        input_ids=all_input_ids,
        input_mask=all_input_mask,
        segment_ids=all_segment_ids,
        label_id=label,
        evidence_start_position=evidence_start_position,
        evidence_end_position=evidence_end_position,
    )


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer,
    output_file, max_qa_length, is_training=False):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)
  all_features = []

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, len(label_list),
                                     max_seq_length, tokenizer, max_qa_length, is_training)
    all_features.append(feature)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_str_feature(values):
      f = tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["example_id"] = create_int_feature([ex_index])
    features["input_ids"] = create_int_feature(sum(feature.input_ids, []))
    features["input_mask"] = create_int_feature(sum(feature.input_mask, []))
    features["segment_ids"] = create_int_feature(sum(feature.segment_ids, []))
    if is_training:
      features["label_ids"] = create_int_feature([feature.label_id])
      features["evidence_start_positions"] = create_int_feature([feature.evidence_start_position])
      features["evidence_end_positions"] = create_int_feature([feature.evidence_end_position])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()

  return all_features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)

#
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, 
                 label_ids, evidence_start_positions, evidence_end_positions,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  num_labels = 4
  bsz_per_core = tf.shape(input_ids)[0]
  max_seq_length = FLAGS.max_seq_length

  input_ids = tf.reshape(input_ids, [bsz_per_core * num_labels, max_seq_length])
  input_mask = tf.reshape(input_mask, [bsz_per_core * num_labels, max_seq_length])
  segment_ids = tf.reshape(segment_ids, [bsz_per_core * num_labels, max_seq_length])

  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_pooled_output()
  output_layer_shape = modeling.get_shape_list(output_layer, expected_rank=2)
  hidden_size = output_layer_shape[1]

  final_hidden = tf.reshape(model.get_sequence_output(), [bsz_per_core, num_labels, max_seq_length, hidden_size])
  final_hidden = tf.reduce_mean(final_hidden, axis=1)
  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/squad/output_weights", [1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [1], initializer=tf.zeros_initializer())

  output_evidence_weights = tf.get_variable(
      "cls/squad/output_evidence_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_evidence_bias = tf.get_variable(
      "cls/squad/output_evidence_bias", [2], initializer=tf.zeros_initializer())

  logits = tf.matmul(output_layer, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)
  logits = tf.reshape(logits, [bsz_per_core, num_labels])
  probabilities = tf.nn.softmax(logits, axis=-1)
  predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
  log_probs = tf.nn.log_softmax(logits, axis=-1)

  if is_training:
    one_hot_labels = tf.one_hot(label_ids, depth=tf.cast(num_labels, dtype=tf.int32), dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    classification_loss = tf.reduce_mean(per_example_loss)
  else:
    per_example_loss = 0
    classification_loss = 0

  def compute_loss(logits, positions):
    one_hot_pos   = tf.one_hot(positions, depth=seq_length, dtype=tf.float32)
    log_probs     = tf.nn.log_softmax(logits, axis=-1)
    loss          = -tf.reduce_mean(tf.reduce_sum(one_hot_pos * log_probs, axis=-1))
    return loss

  final_hidden_matrix = tf.reshape(final_hidden, [batch_size * seq_length, hidden_size])
  evidence_logits = tf.matmul(final_hidden_matrix, output_evidence_weights, transpose_b=True)
  evidence_logits = tf.nn.bias_add(evidence_logits, output_evidence_bias)

  evidence_logits = tf.reshape(evidence_logits, [batch_size, seq_length, 2])
  evidence_logits = tf.transpose(evidence_logits, [2, 0, 1])

  unstacked_evidence_logits = tf.unstack(evidence_logits, axis=0)

  (evidence_start_logits, evidence_end_logits) = (unstacked_evidence_logits[0], unstacked_evidence_logits[1])

  if is_training:
    evidence_start_loss  = compute_loss(evidence_start_logits, evidence_start_positions)
    evidence_end_loss    = compute_loss(evidence_end_logits, evidence_end_positions)
    total_loss = float(FLAGS.loss_lambda) * (evidence_start_loss + evidence_end_loss) / 2 + classification_loss
  else:
    total_loss = -1

  return (total_loss, per_example_loss, probabilities, logits, predictions, evidence_start_logits, evidence_end_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    example_id = features["example_id"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    if is_training:
      label_ids = features["label_ids"]
      evidence_start_positions = features["evidence_start_positions"]
      evidence_end_positions   = features["evidence_end_positions"]
    else:
      label_ids = None
      evidence_start_positions = None
      evidence_end_positions   = None

    (total_loss, per_example_loss, probabilities, logits, predictions, evidence_start_logits, evidence_end_logits) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        evidence_start_positions=evidence_start_positions, 
        evidence_end_positions=evidence_end_positions,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
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
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "example_id": example_id,
          "probabilities": probabilities,
          "predictions": predictions,
          "start_logits": evidence_start_logits,
          "end_logits": evidence_end_logits,
      }
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder, multiple=1):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "example_id": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
  }

  if is_training:
    name_to_features["label_ids"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["evidence_start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["evidence_end_positions"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case):
  """Write final predictions to the json file and log-odds of null if needed."""
  max_answer_length = 200
  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["start_index", "end_index", "start_logit", "end_logit"])

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    prelim_predictions = []
    feature = all_features[example_index] 
    result = all_results[example_index]
    start_indexes = _get_best_indexes(result['start_logits'], n_best_size)
    end_indexes = _get_best_indexes(result['end_logits'], n_best_size)
    for start_index in start_indexes:
      for end_index in end_indexes:
        # We could hypothetically create invalid predictions, e.g., predict
        # that the start of the span is in the question. We throw out all
        # invalid predictions.
        if start_index >= len(feature.tokens[0]):
          continue
        if end_index >= len(feature.tokens[0]):
          continue
        if start_index not in feature.token_to_orig_map:
          continue
        if end_index not in feature.token_to_orig_map:
          continue
        if end_index < start_index:
          continue
        length = end_index - start_index + 1
        if length > max_answer_length:
          continue
        prelim_predictions.append(
            _PrelimPrediction(
                start_index=start_index,
                end_index=end_index,
                start_logit=result['start_logits'][start_index],
                end_logit=result['end_logits'][end_index]))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      if pred.start_index > 0:  # this is a non-null prediction
        tok_tokens = feature.tokens[0][pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[pred.start_index] - 1  # because we have [CLS] at front
        orig_doc_end = feature.token_to_orig_map[pred.end_index] - 1      # because we have [CLS] at front
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = "".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        if FLAGS.task_name == "race":
          orig_text = " ".join(orig_tokens)
        elif FLAGS.task_name == "c3":
          orig_text = "".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case)
        #final_text = final_text.replace(' ','')
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit,
              start_index=pred.start_index,
              end_index=pred.end_index))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=0, end_index=0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry.text
    if best_non_null_entry is None:
      best_non_null_entry = ''

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      output["start_index"] = entry.start_index
      output["end_index"] = entry.end_index
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    all_predictions[example.example_id] = best_non_null_entry
    all_nbest_json[example.example_id] = nbest_json

  return all_predictions


def get_final_text(pred_text, orig_text, do_lower_case):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = "".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose_logging:
      tf.logging.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose_logging:
      tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)
  
  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `do_train` or `do_predict` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  processor = RaceProcessor(
      use_spm=False,
      do_lower_case=FLAGS.do_lower_case)
  label_list = processor.get_labels()
  task_name = 'race'

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=2,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.train_file)
    train_examples_len = len(train_examples)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(int(FLAGS.rand_seed))
    rng.shuffle(train_examples)

    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.
    if not tf.gfile.Exists(FLAGS.train_tfrecord):
      file_based_convert_examples_to_features(
        train_examples,
        label_list,
        FLAGS.max_seq_length,
        tokenizer,
        FLAGS.train_tfrecord,
        FLAGS.max_query_length,
        is_training=True)
    
    num_train_steps = int(train_examples_len / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", train_examples_len)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  # do training
  if FLAGS.do_train:
    train_input_fn = input_fn_builder(
        input_file=FLAGS.train_tfrecord,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        multiple=len(label_list))
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  # do predictions
  if FLAGS.do_predict:
    eval_examples = processor.get_dev_examples(FLAGS.predict_file)
    actual_eval_examples_len = len(eval_examples)

    if FLAGS.use_tpu:
    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on. These do NOT count towards the metric (all tf.metrics
    # support a per-instance weight, and these get a weight of 0.0).
      padding_num = 0
      while len(eval_examples) % FLAGS.predict_batch_size != 0:
        eval_examples.append(PaddingInputExample())
        padding_num += 1

    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.
    all_features = file_based_convert_examples_to_features(
        eval_examples,
        label_list,
        FLAGS.max_seq_length,
        tokenizer,
        FLAGS.predict_tfrecord,
        FLAGS.max_query_length)

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Wring tfrecords into %s", FLAGS.predict_tfrecord)
    tf.logging.info("  Num orig examples = %d (%d actual, %d padding)", len(eval_examples), actual_eval_examples_len, padding_num)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = input_fn_builder(
        input_file=FLAGS.predict_tfrecord,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False,
        multiple=len(label_list))

    def pred_to_option(x):
      return chr(ord('A')+int(x))

    # If running eval on the TPU, you will need to specify the number of steps.
    all_results = []
    predicted_num = 0
    for result in estimator.predict(predict_input_fn, yield_single_examples=True):
      if len(all_results) % 100 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      if predicted_num < actual_eval_examples_len:
        all_results.append(result)
      predicted_num += 1

    eval_examples = eval_examples[:actual_eval_examples_len]
    all_evidences = write_predictions(eval_examples, all_features, all_results, 
                                      FLAGS.n_best_size, FLAGS.max_answer_length, FLAGS.do_lower_case)
    
    all_answers_evidences = dict()
    for ans_i in range(len(eval_examples)):
      qid = eval_examples[ans_i].example_id
      temp_result = all_results[ans_i]
      temp_answer = pred_to_option(temp_result['predictions'])
      temp_evidence = all_evidences[qid]
      all_answers_evidences[qid] = {'answer': temp_answer, 'evidence': temp_evidence}
    
    output_prediction_file = os.path.join(FLAGS.output_dir, "dev_predictions.json")
    tf.logging.info("Writing Predictions to %s" % output_prediction_file)
    with tf.gfile.GFile(output_prediction_file, "w") as pred_writer:
      tf.logging.info("***** Writing Predictions *****")
      pred_writer.write(json.dumps(all_answers_evidences, indent=2, ensure_ascii=False)+"\n")

if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
