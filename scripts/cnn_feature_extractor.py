import os

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import string_ops
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import vgg_preprocessing
import numpy as np
from tqdm import tqdm


slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    "checkpoint_dir",
    "./resnet_v1_101.ckpt",
    "Resnet checkpoint to use"
)
tf.app.flags.DEFINE_string(
    "image_dir",
    "../data/images/",
    ""
)
tf.app.flags.DEFINE_string(
    "input_fname",
    "../data/caption_dataset/train.txt",
    ""
)
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use")
tf.app.flags.DEFINE_string("gpu_id", "0", "GPU id to use")
tf.app.flags.DEFINE_string(
    "output_dir",
    "../data/resnet_pool5_features/",
    "Output directory to save resnet features"
)


def decode_image(contents, channels=None, name=None):
  """Convenience function for `decode_gif`, `decode_jpeg`, and `decode_png`.
  Detects whether an image is a GIF, JPEG, or PNG, and performs the appropriate
  operation to convert the input bytes `string` into a `Tensor` of type `uint8`.

  Note: `decode_gif` returns a 4-D array `[num_frames, height, width, 3]`, as
  opposed to `decode_jpeg` and `decode_png`, which return 3-D arrays
  `[height, width, num_channels]`. Make sure to take this into account when
  constructing your graph if you are intermixing GIF files with JPEG and/or PNG
  files.

  Args:
    contents: 0-D `string`. The encoded image bytes.
    channels: An optional `int`. Defaults to `0`. Number of color channels for
      the decoded image.
    name: A name for the operation (optional)

  Returns:
    `Tensor` with type `uint8` with shape `[height, width, num_channels]` for
      JPEG and PNG images and shape `[num_frames, height, width, 3]` for GIF
      images.
  """
  with ops.name_scope(name, 'decode_image'):
    if channels not in (None, 0, 1, 3):
      raise ValueError('channels must be in (None, 0, 1, 3)')
    substr = string_ops.substr(contents, 0, 4)

    def _png():
      return gen_image_ops.decode_png(contents, channels)

    def _jpeg():
      return gen_image_ops.decode_jpeg(contents, channels)

    is_png = math_ops.equal(substr, b'\211PNG', name='is_png')
    return control_flow_ops.cond(is_png, _png, _jpeg, name='cond_jpeg')


def main(_):
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  with tf.Graph().as_default() as g:
    with open(FLAGS.input_fname, 'r') as f:
      filenames = [line.split(',')[0][:-4] for line in f.readlines()]
      filenames = [
          os.path.join(FLAGS.image_dir, name) for name in filenames \
              if not os.path.exists(os.path.join(FLAGS.output_dir, name + '.npy'))
      ]

    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value, channels=3)
    image_size = resnet_v1.resnet_v1.default_image_size
    processed_image = vgg_preprocessing.preprocess_image(
        image, image_size, image_size, is_training=False
    )
    processed_images, keys = tf.train.batch(
        [processed_image, key],
        FLAGS.batch_size,
        num_threads=8, capacity=8*FLAGS.batch_size*5,
        allow_smaller_final_batch=True
    )

    # Create the model
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(
          processed_images, num_classes=1000, is_training=False
      )
      init_fn = slim.assign_from_checkpoint_fn(
          FLAGS.checkpoint_dir, slim.get_model_variables()
      )
      pool5 = g.get_operation_by_name('resnet_v1_101/pool5').outputs[0]
      pool5 = tf.transpose(pool5, perm=[0, 3, 1, 2])  # (batch_size, 2048, 1, 1)

      with tf.Session() as sess:
        init_fn(sess)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
          for step in tqdm(
              xrange(len(filenames) / FLAGS.batch_size + 1), ncols=70
          ):
            if coord.should_stop():
              break
            file_names, pool5_value = sess.run([keys, pool5])
            for i in xrange(len(file_names)):
              np.save(
                  os.path.join(
                      FLAGS.output_dir,
                      os.path.basename(file_names[i]) + '.npy'
                  ),
                  pool5_value[i].astype(np.float32)
              )
        except tf.errors.OutOfRangeError:
          print "Done feature extraction -- epoch limit reached"
        finally:
          coord.request_stop()

        coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
