from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
# tensorflow의 로깅하는 기능과 모니터링할 때 사용하는 api
# 로깅을 하지 않으면 실제 어덯게 동작하고 수행이 되고 있는지에 대해서
# 우리가 알 수 없다. 기본적인 테스트를 진행할 때 적절한 로깅은 필수
# 수렴 과정이 다 진행되지 않았는데도 불구하고 완료로 처리된 것은 아닌지 등을
# 확인해야 하기 때문에 학습 과정에서 로깅을 해야 한다.
# Tensorflow는 5가지 타입의 로깅을 제공한다. DEBUG, INFO,WARN,ERROR,FATAL
# 순차적으로 상위의 로깅 타입이 되며, 만약 로깅 설정시에 ERROR로 되어 있으면
# ERROR와 FATAL 로그를 볼 수 있다. DEBUG로 설정하면 모든 타입의 로그를 볼 수 있다.
# 이렇게 심플하게 LOSS를 확인하는 것도 좋지만, Monitor API를 활용하는 것이 좋다.
# Monitor API는 CaptureVariable, PrintTensor, SummarySaver, ValidationMonitor가 있다.

def cnn_model_fn(features, labels, mode):

  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # melspectrograms are 60x41 pixels, and have 2 channel
  input_layer = tf.cast(tf.convert_to_tensor(features["x"]),tf.float32)
  print(labels)
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=80,
      kernel_size=[57, 6],
      strides = [1,1],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 3], strides=[1,3])
  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=80,
      kernel_size=[1, 3],
      strides=[1,1],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 3], strides=[1,3])
  # Flatten tensor into a batch of vectors
  pool2_flat = tf.reshape(pool2, [-1, 57 * 11 * 80])

  # Dense Layer
  dense1 = tf.layers.dense(inputs=pool2_flat, units=5000, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  dense2 = tf.layers.dense(inputs=dropout1, units=5000, activation=tf.nn.relu)
  dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(inputs=dropout2, units=10)

  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  # return EstimatorSpec
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  print(onehot_labels)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):

    train_labels = 0
    train_features = 0
    test_labels = 0
    test_features = 0

    # Load training and eval data
    for i in range(1, 4):
        # Extraction / audioaudioextraction_90_0.npz
        tr = "Extraction/melspectrogram_90_" + str(i) + ".npz"
        tr_data = np.load(tr)

        tr_features = tr_data["features"]
        tr_labels = tr_data["labels"]
        tr_labels = np.array(tr_labels,np.int)

        if i == 1:
            train_features = tr_features
            train_labels = tr_labels
        else:
            train_features = np.append(train_features, tr_features,axis=0)
            train_labels = np.append(train_labels, tr_labels,axis=0)
        # features=features,labels=labels
        print(train_labels.shape)

    print("Training dataset load finish!")
    for i in range(9, 11):
        ts = "Extraction/melspectrogram_90_" + str(i) + ".npz"
        ts_data = np.load(ts)

        ts_features = ts_data["features"]
        ts_labels = ts_data["labels"]
        ts_labels = np.array(ts_labels, np.int)
        if i == 9:
            test_features = ts_features
            test_labels = ts_labels
        else:
            test_features = np.append(test_features, ts_features, axis=0)
            test_labels = np.append(test_labels, ts_labels, axis=0)
        print(test_labels.shape)

    print("Test dataset load finish!")

    sound_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir="/tmp/urbansound_model")
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": train_features},
        y = train_labels,
        batch_size = 1000,
        num_epochs = None,
        shuffle = True)

    sound_classifier.train(
            input_fn = train_input_fn,
            steps = 2000,
            hooks = [logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x" : test_features},
        y = test_labels,
        num_epochs = 1,
        shuffle = False)

    eval_results = sound_classifier.evalutate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
  tf.app.run()