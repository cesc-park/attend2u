from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import colorlog
import time
from utils.data_utils import enqueue
from utils.configuration import ModelConfig
from model.model import CSMN
from scripts.generate_dataset import EOS_ID
from utils.evaluator import Evaluator
from termcolor import colored
flags = tf.app.flags

flags.DEFINE_string('eval_dir', './checkpoints/eval',
                           """Directory where to write event logs.""")
flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
flags.DEFINE_string("train_dir", "./checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_string(
    "vocab_fname",
    "./data/caption_dataset/40000.vocab",
    "Vocabulary file for evaluation"
)
flags.DEFINE_integer("num_gpus", 4, "Number of gpus to use")
flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                            """How often to run the eval.""")
flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
TOWER_NAME = 'tower'



FLAGS = flags.FLAGS

def _load_vocabulary(vocab_fname):
  with open(vocab_fname, 'r') as f:
    vocab = f.readlines()
  vocab = [s.strip() for s in vocab]
  rev_vocab = {}
  for i, token in enumerate(vocab):
    rev_vocab[i] = token
  return vocab, rev_vocab

def _inject_summary(key_value):
    summary = tf.Summary()
    for key, value in key_value.iteritems():
      summary.value.add(tag='%s' % (key), simple_value=value)
    return summary

def _eval_once(saver, summary_writer, argmaxs, answer_ids, vocab, rev_vocab,
    num_examples_per_epoch, b_global_step):
  """Run Eval once.
  """
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)

      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      if global_step == b_global_step:
          return global_step
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      num_iter = 1 + int(
          num_examples_per_epoch / FLAGS.batch_size / FLAGS.num_gpus
      )


      desc_list = []
      answer_list = []
      desc_token_list = []
      answer_token_list = []
      step = 0
      while step < num_iter and not coord.should_stop():
        results = sess.run([argmaxs, answer_ids])
        desc = results[0].tolist()
        answer = results[1].tolist()
        desc_list += desc
        answer_list += answer
        step += 1

      for i in xrange(len(desc_list)):
        desc = []
        answer = []
        for k in xrange(len(desc_list[i])):
          token_id = desc_list[i][k]
          if token_id == EOS_ID:
            break
          desc.append(rev_vocab[token_id])
        for k in xrange(len(answer_list[i])):
          token_id = answer_list[i][k]
          if token_id == EOS_ID:
            break
          answer.append(rev_vocab[token_id])
        desc_token_list.append(desc)
        answer_token_list.append(answer)

      colorlog.info(
          colored("Validation Output Example (%s)" % global_step, 'green')
      )
      for i, (desc, answer) in enumerate(
          zip(desc_token_list[:15], answer_token_list[:15])
      ):
        print("%d." % (i))
        print(' '.join(answer))
        print(' '.join(desc) + "\n")

      evaluator = Evaluator()
      result = evaluator.evaluation(
          desc_token_list, answer_token_list, "coco"
      )

      summary = _inject_summary(result)
      summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
  return global_step



def evaluate():
  # Read vocabulary
  vocab, rev_vocab = _load_vocabulary(FLAGS.vocab_fname)

  with tf.Graph().as_default() as g:
    #Enque data for evaluation
    num_examples_per_epoch, tower_img_embedding, tower_context_length, \
        tower_caption_length, tower_context_id, tower_caption_id, \
        tower_answer_id, tower_context_mask, \
        tower_caption_mask = enqueue(True)

    tower_argmax = []
    # Calculate the gradients for each model tower.
    with tf.variable_scope(tf.get_variable_scope()) as scope:
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
            inputs = [
                tower_img_embedding[i],
                tower_context_length[i],
                tower_caption_length[i],
                tower_context_id[i],
                tower_caption_id[i],
                tower_answer_id[i],
                tower_context_mask[i],
                tower_caption_mask[i]
            ]
            net = CSMN(inputs, ModelConfig(FLAGS), is_training= False)
            argmax = net.argmax
            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Keep track of the gradients across all towers.
            tower_argmax.append(argmax)
    argmaxs = tf.concat(tower_argmax, 0)
    answer_ids = tf.concat(tower_answer_id, 0)
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    #Don't evaluate again for the same checkpoint.
    b_g_s = "0"
    while True:
      c_g_s = _eval_once(
          saver, summary_writer, argmaxs, answer_ids, vocab,
          rev_vocab, num_examples_per_epoch, b_g_s
      )
      b_g_s = c_g_s
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
  if not tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
