import tensorflow as tf
print("GPUs:", len(tf.config.list_physical_devices('GPU')))
