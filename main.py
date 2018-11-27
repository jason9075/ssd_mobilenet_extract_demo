import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def show_tensor_by_ckpt(file_name):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()

    for key in var_to_shape_map:
        print("tensor_name: ", key)


def main():
    graph = tf.Graph()

    with tf.Session(graph=graph) as sess:
        tf.Variable([1], tf.float32)

        # show_tensor_by_ckpt('models/ssd_mobilenet_v2_coco_2018_03_29/models.ckpt')
        saver = tf.train.import_meta_graph('models/ssd_mobilenet_v2_coco_2018_03_29/models.ckpt.meta')
        saver.restore(sess, 'models/ssd_mobilenet_v2_coco_2018_03_29/models.ckpt')

        # tf.summary.FileWriter("tensor_board/", graph=graph)


if __name__ == '__main__':
    main()
