# coding=utf-8

from utils import run_lib_flowgrad
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Rectified Flow Model configuration.", lock_config=True)
flags.DEFINE_enum("mode", 'flowgrad-edit', ["flowgrad-edit"], "Running mode.")
flags.DEFINE_string("text_prompt", None, "text prompt for editing")
flags.DEFINE_float("alpha", 0.7, "The coefficient to balance the edit loss and the reconstruction loss.")
flags.DEFINE_string("model_path", None, "Path to pre-trained model checkpoint.")
flags.DEFINE_string("image_path", None, "The path to the image that will be edited")
flags.DEFINE_string("output_folder", "output", "The folder name for storing output")
flags.mark_flags_as_required(["model_path", "text_prompt", "alpha", "image_path"])


def main(argv):
  if FLAGS.mode == "flowgrad-edit":
    run_lib_flowgrad.flowgrad_edit(FLAGS.config, FLAGS.text_prompt, FLAGS.alpha, FLAGS.model_path, FLAGS.image_path, FLAGS.output_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
