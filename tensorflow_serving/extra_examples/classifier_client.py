"""
Send preprocessed image to inference server for classification. This client assumes an 
inception-v3 style classification pipeline where the image is centered, normalized and 
resized to 299x299x3. 
"""

from grpc.beta import implementations
import numpy as np
from scipy.misc import imread, imresize
import sys
import threading
import tensorflow as tf
import time

import services_pb2

tf.app.flags.DEFINE_string('server', 'localhost:9000', 'inference service host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = services_pb2.beta_create_ClassificationService_stub(channel)
  
  # Read in and format the image
  image = imread(FLAGS.image)
  image = imresize(image, (299, 299, 3)).astype(np.float)
  image -= 128.
  image /= 128.
  
  # Construct the request
  request = services_pb2.ClassificationRequest()
  request.image_data.extend(image.ravel().tolist())
  
  # Send the request to the server
  t = time.time()  
  result = stub.Classify(request, 10.0)  # 10 secs timeout
  dt = time.time() - t
  
  print result
  print "Image classification time (ms): %.1f" % (dt * 1000)

if __name__ == '__main__':
  tf.app.run()
