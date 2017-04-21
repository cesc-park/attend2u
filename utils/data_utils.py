from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import random
from scripts.generate_dataset import GO_ID, EOS_ID
flags = tf.app.flags
# Basic model parameters.
flags.DEFINE_string("data_dir",
    "./data/caption_dataset",
    "data directory [data]"
)
flags.DEFINE_string("img_data_dir",
    "./data/resnet_pool5_features",
    "data directory [data]"
)

flags.DEFINE_integer("max_context_length", 60,
    "User contex max length default [60]"
)
flags.DEFINE_integer("max_output_length", 16,
    "User contex max length default [60]"
)
flags.DEFINE_integer('batch_size', 200,
    """Number of examples to process in a batch."""
)
flags.DEFINE_integer('vocab_size', 40000,
    """Number of vocab."""
)
flags.DEFINE_integer('word_emb_dim', 512,
    """Dimensions of word embeddings."""
)
flags.DEFINE_integer('mem_dim', 1024,
    """Dimensions of memories."""
)
flags.DEFINE_integer('num_channels', 300,
    """Number of channels of memory cnn."""
)







FLAGS = flags.FLAGS


root_path = "/"
train_fpath = 'train.txt'
val_fpath = 'test1.txt'


def _generate_data_and_label_batch(inputs, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch

  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' data from the example queue.
  num_preprocess_threads = 8
  if shuffle:
    outputs = tf.train.shuffle_batch(
        inputs,
        batch_size=1,
        num_threads=num_preprocess_threads,
        enqueue_many=False,
        capacity=min_queue_examples + 3 ,
        min_after_dequeue=min_queue_examples
    )
  else:
    outputs = tf.train.batch(
        inputs,
        batch_size=1,
        num_threads=num_preprocess_threads,
        enqueue_many=False,
        capacity=min_queue_examples + 3
    )

  return outputs

def numpy_read_func(batch_path):
  np_list = []
  for i in range(len(batch_path)):
    np_list.append(
        np.load(os.path.join(FLAGS.img_data_dir, batch_path[i]))
    )
  return np.array(np_list)

def token_split_func(batch_token, max_length, cap=None):
  all_ids = np.zeros([FLAGS.batch_size, max_length], dtype=np.int32)
  for i in range(FLAGS.batch_size):
    if "_" in batch_token[i]:
      valid_ids = map(int, batch_token[i].split('_'))[:max_length]
    elif len(batch_token[i]) == 0:
      valid_ids = []
    else:
      valid_ids = [int(batch_token[i])]
    if cap:
      valid_ids = valid_ids[:max_length-1]
      if cap == 'caption' :
        all_ids[i, :len(valid_ids)+1]= [GO_ID] + valid_ids[:]
      else:
        all_ids[i, :len(valid_ids)+1]= valid_ids[:] + [EOS_ID]
    else:
      all_ids[i, :len(valid_ids)] = valid_ids[:]
  return all_ids

def mask_build_func(batch_length, max_length, cap=None):
  mask = np.zeros([FLAGS.batch_size, max_length], dtype=np.bool)
  for i in range(FLAGS.batch_size):
    if cap:
      length = int(batch_length[i]) + 1
    else:
      length = int(batch_length[i])
    length = max_length if length > max_length else length
    mask[i, :length] = True
  return mask

def read_numpy_format_and_label(filename_queue):
  # Tricks to make the batch have
  # almost same caption length
  # We should group it by using dequeue_many and enqueue many
  # from sorted list.
  filename_and_label_tensor = filename_queue.dequeue_many(
      FLAGS.batch_size
  )
  batch_filename, batch_context_length, batch_caption_length, \
      batch_context, batch_caption = tf.decode_csv(
          filename_and_label_tensor,
          [[""], [""], [""], [""], [""]]
      )
  batch_context_length = tf.minimum(
      tf.cast(
          tf.string_to_number(batch_context_length),
          tf.int32
      ),
      FLAGS.max_context_length
  )
  batch_caption_length = tf.minimum(
      tf.cast(
          tf.string_to_number(batch_caption_length),
          tf.int32
      )+1,
      FLAGS.max_output_length
  )
  batch_img_embedding = tf.py_func(
      numpy_read_func,
      [batch_filename],
      tf.float32
  )
  batch_context_id = tf.py_func(
      token_split_func,
      [batch_context, FLAGS.max_context_length],
      tf.int32
  )
  batch_caption_id = tf.py_func(
      token_split_func,
      [batch_caption, FLAGS.max_output_length, 'caption'],
      tf.int32
  )
  batch_answer_id = tf.py_func(
      token_split_func,
      [batch_caption, FLAGS.max_output_length, 'answer'],
      tf.int32
  )
  batch_context_mask = tf.py_func(
      mask_build_func,
      [batch_context_length, FLAGS.max_context_length],
      tf.bool
  )
  batch_caption_mask = tf.py_func(
      mask_build_func,
      [batch_caption_length, FLAGS.max_output_length, 'caption'],
      tf.bool
  )

  return batch_img_embedding, batch_context_length, \
      batch_caption_length, batch_context_id, batch_caption_id, \
      batch_answer_id, batch_context_mask, batch_caption_mask
def chunks(data, batch_size):
  chunk_l = []
  N = int(len(data)/batch_size)
  for i in xrange(N):
    chunk_l.append(data[i*batch_size:(i+1)*batch_size])
  chunk_l.append(data[(i+1)*batch_size:])
  return chunk_l

def mergers(chunk_l):
  newdata = []
  for datas in chunk_l:
    for data in datas:
      newdata.append(data)
  return newdata
def enqueue(eval_data):

  # string input format
  # numpyfname,contextlength,captionlength,contexttoken1_contexttoken2,wordtoken1_wordtoken2
  # e.g. 12345.npy,4,3,445_24_445_232,134_466_234
  if not eval_data:
    filenames = [l.strip() for l in open(os.path.join(FLAGS.data_dir, train_fpath)).readlines()]
    chunk_list = chunks(filenames, FLAGS.batch_size)
    random.shuffle(chunk_list)
    filenames = mergers(chunk_list)
  else:
    filenames = [l.strip() for l in open(os.path.join(FLAGS.data_dir, val_fpath)).readlines()]
  num_examples_per_epoch = len(filenames)

  # Create a queue that produces the filenames to read.
  # Don't shuffle.. to maintain the order
  # Data will be shuffled in the process of the tf.train.shuffle_batch
  filename_queue = tf.train.string_input_producer(
    filenames,
    shuffle=False,
    capacity=num_examples_per_epoch
  )


  img_embedding_list = []
  context_length_list = []
  caption_length_list = []
  context_id_list = []
  caption_id_list = []
  answer_id_list = []
  context_mask_list = []
  caption_mask_list = []

  for it in range(FLAGS.num_gpus):
    img_embedding, context_length, caption_length, context_id, caption_id, \
        answer_id, context_mask, caption_mask = read_numpy_format_and_label( filename_queue)

    # Set the shapes of tensors
    img_embedding.set_shape([FLAGS.batch_size, 2048,1,1])
    img_embedding = tf.transpose(
        tf.reshape(img_embedding, [FLAGS.batch_size, 2048,1]),
        perm=[0, 2, 1]
    )
    context_length.set_shape([FLAGS.batch_size])
    caption_length.set_shape([FLAGS.batch_size])
    context_id.set_shape([FLAGS.batch_size, FLAGS.max_context_length])
    caption_id.set_shape([FLAGS.batch_size, FLAGS.max_output_length])
    answer_id.set_shape([FLAGS.batch_size, FLAGS.max_output_length])
    context_mask.set_shape([FLAGS.batch_size, FLAGS.max_context_length])
    caption_mask.set_shape([FLAGS.batch_size, FLAGS.max_output_length])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.01
    min_queue_examples = int(
        num_examples_per_epoch * min_fraction_of_examples_in_queue / FLAGS.batch_size
    )

    # Generate a batch of images and labels by building up a queue of examples.
    img_embeddings, context_lengths, caption_lengths, \
        context_ids, caption_ids, answer_ids, context_masks, \
        caption_masks = _generate_data_and_label_batch(
            [img_embedding, context_length, caption_length, \
            context_id, caption_id, answer_id, context_mask, caption_mask],
            min_queue_examples, FLAGS.batch_size,
            shuffle=(not eval_data)
        )
    img_embeddings = tf.squeeze(img_embeddings, axis = [0])
    context_lengths = tf.squeeze(context_lengths, axis = [0])
    caption_lengths = tf.squeeze(caption_lengths, axis = [0])
    context_ids = tf.squeeze(context_ids, axis = [0])
    caption_ids = tf.squeeze(caption_ids, axis = [0])
    answer_ids = tf.squeeze(answer_ids, axis = [0])
    context_masks = tf.squeeze(context_masks, axis = [0])
    caption_masks = tf.squeeze(caption_masks, axis = [0])

    # Save tensors
    img_embedding_list.append(img_embeddings)
    context_length_list.append(context_lengths)
    caption_length_list.append(caption_lengths)
    context_id_list.append(context_ids)
    caption_id_list.append(caption_ids)
    answer_id_list.append(answer_ids)
    context_mask_list.append(context_masks)
    caption_mask_list.append(caption_masks)

  return num_examples_per_epoch, img_embedding_list, context_length_list, \
      caption_length_list, context_id_list, caption_id_list, answer_id_list, \
      context_mask_list, caption_mask_list
