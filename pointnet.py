import os
import glob
import config
import trimesh
import numpy as np
import tensorflow as tf
from model import create_pointnet_model
from matplotlib import pyplot as plt




#############
## Uncomment to show sample mesh and the point cloud generated from it
# mesh = trimesh.load(os.path.join(config.DATA_DIR, "sofa/train/sofa_0004.off"))
# mesh.show()

# points = mesh.sample(2048)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:, 0], points[:, 1], points[:, 2])
# ax.set_axis_off()
# plt.show()
#############


def parse_dataset(num_points=2048):

	print("[INFO] ---- Started Processing the Dataset")
	train_points = []
	train_labels = []
	test_points = []
	test_labels = []

	class_map = dict()

	folders = glob.glob(os.path.join(config.DATA_DIR,"[!R]*"))
    
	for i, folder in enumerate(folders):
		class_name = os.path.basename(folder)
		print("[INFO] ---- Processing Class: {}".format(class_name))

		class_map[i] = class_name

		train_files = glob.glob(os.path.join(folder, "train/*"))
		test_files = glob.glob(os.path.join(folder, "test/*"))

		for f in train_files:
		    train_points.append(trimesh.load(f).sample(num_points))
		    train_labels.append(i)
		    
		for f in test_files:
		    test_points.append(trimesh.load(f).sample(num_points))
		    test_labels.append(i)

	return np.array(train_points), np.array(test_points), np.array(train_labels), np.array(test_labels), class_map

def augment(points, label):
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    points = tf.random.shuffle(points)
    return points, label

def evaluate_visualize_trained_model(model, test_dataset):
	data = test_dataset.take(1)

	points, labels = list(data)[0]
	points = points[:8, ...]
	labels = labels[:8, ...]

	# run test data through model
	preds = model.predict(points)
	preds = tf.math.argmax(preds, -1)
	points = points.numpy()

	# plot points with predicted class and label
	fig = plt.figure(figsize=(15, 10))
	for i in range(8):
	    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
	    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
	    ax.set_title(f"label: {CLASS_MAP[labels.numpy()[i]]}, pred: {CLASS_MAP[preds[i].numpy()]}")
	    ax.set_axis_off()
	plt.show()

if __name__ == '__main__':

	train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(config.NUM_POINTS)
	print(CLASS_MAP)

	train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
	test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

	train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(config.BATCH_SIZE)
	test_dataset = test_dataset.shuffle(len(test_points)).batch(config.BATCH_SIZE)

	model = create_pointnet_model(config.NUM_POINTS, config.NUM_CLASSES)
	model.summary()

	model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
    metrics=["sparse_categorical_accuracy"]
    )

	model.fit(train_dataset, epochs=config.NUM_EPOCHS, validation_data=test_dataset)

	evaluate_visualize_trained_model(model, test_dataset)


