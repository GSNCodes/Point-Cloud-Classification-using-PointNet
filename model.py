import tensorflow as tf
import numpy as np

def conv_bn(x, filters):
    x = tf.keras.layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
    return tf.keras.layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = tf.keras.layers.Dense(filters)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
    return tf.keras.layers.Activation("relu")(x)

class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        config = {"num_features": self.num_features, "l2reg": self.l2reg}
        return config


def tnet(inputs, num_features):

    bias = tf.keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 64)
    x = conv_bn(x, 128)
    x = conv_bn(x, 1024)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512)
    x = dense_bn(x, 256)
    x = tf.keras.layers.Dense(
    	num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg
        )(x)

    feat_T = tf.keras.layers.Reshape((num_features, num_features))(x)


    return tf.keras.layers.Dot(axes=(2, 1))([inputs, feat_T])


def create_pointnet_model(NUM_POINTS, NUM_CLASSES):
	inputs = tf.keras.Input(shape=(NUM_POINTS, 3))

	x = tnet(inputs, 3)
	x = conv_bn(x, 64)
	x = conv_bn(x, 64)
	x = tnet(x, 64)
	x = conv_bn(x, 64)
	x = conv_bn(x, 128)
	x = conv_bn(x, 1024)
	x = tf.keras.layers.GlobalMaxPooling1D()(x)
	x = dense_bn(x, 512)
	x = tf.keras.layers.Dropout(0.3)(x)
	x = dense_bn(x, 256)
	x = tf.keras.layers.Dropout(0.3)(x)

	outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

	model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

	return model
