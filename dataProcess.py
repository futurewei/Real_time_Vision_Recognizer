import tensorflow as tf
import numpy as np 
import os
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt



N_CLASSES = 2
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 5000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 
train_folder = 'train/'

#label 
def data_processing():
	pen = []
	pen_label = []
	others = []
	others_label = []
	for file in os.listdir(train_folder):
		data = file.split(sep = '.')
		if(len(data[0])>=9 and data[0][:9] == 'n03477512'):
			pen.append(train_folder + file)
			pen_label.append(1)
		else:
			others.append(train_folder+file)
			others_label.append(0)

	image_data = np.hstack((pen, others))
	label_data = np.hstack((pen_label, others_label))
	image_list = image_data.tolist()
	label_list = label_data.tolist()
	return image_list, label_list


def get_batch(images, labels, image_W, image_H, batch_size, shuffle, capacity):
    num_preprocess_threads = 64
    images = tf.cast(images, tf.string)
    labels = tf.cast(labels, tf.int32)
    input_queue = tf.train.slice_input_producer([images, labels])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
	# crop images
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    if shuffle:
    	images_batch, label_batch = tf.train.shuffle_batch(
        	[image, label],
        	batch_size=batch_size,
        	num_threads=num_preprocess_threads,
        	capacity=capacity,
        	min_after_dequeue=capacity-1)
    else:
    	images_batch, label_batch = tf.train.batch(
        	[image, label],
        	batch_size=batch_size,
        	num_threads=num_preprocess_threads,
        	capacity=capacity)
    images_batch = tf.cast(images_batch, tf.float32)
    return images_batch, tf.reshape(label_batch, [batch_size])




def inference(images, batch_size):
  """Build the MNIST model up to where it may be used for inference.
  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  with tf.variable_scope('conv1') as scope:
    weights = tf.get_variable('weights', 
                                  shape = [3,3,3, 16],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
    conv = tf.nn.conv2d(images, weights, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', 
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    weights =tf.get_variable('weights',
                                  shape=[3,3,16,16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
    conv = tf.nn.conv2d(norm1, weights, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases',
                                 shape=[16], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = tf.get_variable('weights',
                                  shape=[dim,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
    biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = tf.get_variable('weights',
                                  shape=[128,128],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
    biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = tf.get_variable('softmax_linear',
                                  shape=[128, N_CLASSES],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
    biases = tf.get_variable('biases', 
                                 shape=[N_CLASSES],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
  return softmax_linear

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
"""
  with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy

def run_training():
  logs_train_dir = 'logs/'

  train, train_label = data_processing()

  train_batch, train_label_batch = get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, False, CAPACITY)
  
  train_logits = inference(train_batch, BATCH_SIZE)
  train_loss = loss(train_logits, train_label_batch)
  train_op = training(train_loss, learning_rate)
  accuracy = evaluation(train_logits, train_label_batch) #?why its still train labels.
  
  summary_op = tf.summary.merge_all()
  sess = tf.Session()
  train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(logs_train_dir)

  sess.run(tf.global_variables_initializer())
  saver.restore(sess, ckpt.model_checkpoint_path)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess = sess, coord =coord)
  
  try:
    for step in np.arange(MAX_STEP):
       if coord.should_stop():
          break
       _, tra_loss, tra_acc = sess.run([train_op, train_loss, accuracy])
       
       if step%50 ==0:
          print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc))
          summary_str = sess.run(summary_op)
          train_writer.add_summary(summary_str, step)

       if step % 100 == 0 or (step + 1) == MAX_STEP:
          checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

  except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
  finally:
    coord.request_stop()

  coord.join(threads)
  sess.close()

def check(name):
    image = Image.open(name)
    #plt.imshow(image)
    image = image.resize([208, 208])
    image_array = np.array(image)
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        
        image = tf.cast(image_array, tf.float32)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = inference(image, BATCH_SIZE,)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
        
        logs_train_dir = 'logs/'
                       
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is others with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a book with possibility %.6f' %prediction[:, 1])
                #if prediction[:,1]>0.9:
                return 'This is a book with possibility %.6f' %prediction[:, 1]
                #return False


