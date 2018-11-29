from sklearn import datasets

dataset = datasets.load_boston()
data, target = dataset.data, dataset.target.reshape(-1, 1)

from sklearn import preprocessing

data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.MinMaxScaler()

data = data_scaler.fit_transform(data)
target = target_scaler.fit_transform(target)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
        data, target, train_size = 0.85
        )

learning_rate = 0.4
epochs = 3000

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 13])
y = tf.placeholder(tf.float32, [None, 1])

y_predict = tf.placeholder(tf.float32, [None, 1])
#y_test    = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([13, 50], stddev = 0.03), name = 'W1')
b1 = tf.Variable(tf.random_normal([50], stddev = 0.03), name = 'b1')

z1 = tf.add(tf.matmul(x, W1), b1)

a1 = tf.nn.relu(z1)

W2 = tf.Variable(tf.random_normal([50, 1], stddev = 0.03), name = 'W2')
b2 = tf.Variable(tf.random_normal([1], stddev = 0.03), name = 'b2')

z2 = tf.add(tf.matmul(a1, W2), b2)

y_ = tf.nn.sigmoid(z2)

quad_loss = tf.reduce_mean(tf.square(y - y_))

optimiser = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(quad_loss)

init_op = tf.global_variables_initializer()

error = tf.sqrt(tf.reduce_mean(tf.square(tf.log(y_predict) - tf.log(y))))

with tf.Session() as sess:
  sess.run(init_op)
  
  for epoch in range(epochs):
      _, c = sess.run([optimiser, quad_loss], 
                          feed_dict = {x: x_train, y: y_train})
      
      print("Epoch", (epoch + 1), ": cost =", "{:.3f}".format(c))
  
  y_p = sess.run(y_, feed_dict = {x: x_test})
  y_predict_scaled = target_scaler.inverse_transform(y_p)
  y_test_scaled = target_scaler.inverse_transform(y_test)
  
  print("Root Mean Squared Log Error: ", sess.run(error, feed_dict = {y_predict: y_predict_scaled, 
                                                y: y_test_scaled}))
  
  #Some Example Predictions compared to actual prices
  print(y_predict_scaled[:5, 0])
  print(y_test_scaled[:5, 0])