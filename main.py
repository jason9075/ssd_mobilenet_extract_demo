import tensorflow as tf
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from tensorflow.python import pywrap_tensorflow


def show_tensor_by_ckpt(file_name):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()

    for key in var_to_shape_map:
        print("tensor_name: ", key)


def display_ui(image, locations, classes, scores):
    result = Image.fromarray(image)

    draw = ImageDraw.Draw(result)
    h, w, _ = image.shape
    font = ImageFont.truetype("fonts/Raleway-Regular.ttf", 24)
    for index, location in enumerate(locations):
        if classes[0][index] != 1:  # filter person
            continue
        if scores[0][index] < 0.65:
            continue
        point_1 = (int(location[1] * w), int(location[0] * h))
        point_2 = (int(location[3] * w), int(location[2] * h))
        draw.rectangle((point_1, point_2), outline='green')
        draw.text(point_1, str(index), font=font, fill='red')

    result.show()


def main():
    graph = tf.Graph()

    # image = Image.open('street.jpg')
    image = Image.open('3_persons.jpg')
    image = np.asarray(image)

    with tf.Session(graph=graph) as sess:
        tf.Variable([1], tf.float32)

        # show_tensor_by_ckpt('models/ssd_mobilenet_v2_coco_2018_03_29/models.ckpt')
        saver = tf.train.import_meta_graph('models/ssd_mobilenet_v2_coco_2018_03_29/model.ckpt.meta')
        saver.restore(sess, 'models/ssd_mobilenet_v2_coco_2018_03_29/model.ckpt')

        # tf.summary.FileWriter("tensor_board/", graph=graph)

        input_image = graph.get_tensor_by_name('image_tensor:0')

        boxes_t = graph.get_tensor_by_name('detection_boxes:0')
        scores_t = graph.get_tensor_by_name('detection_scores:0')
        num_dets_t = graph.get_tensor_by_name('num_detections:0')
        classes_t = graph.get_tensor_by_name('detection_classes:0')

        locations, scores, num_dets, classes = sess.run([boxes_t, scores_t, num_dets_t, classes_t],
                                                        feed_dict={input_image: np.expand_dims(image, 0)})
        locations = locations[0]

        display_ui(image, locations, classes, scores)


if __name__ == '__main__':
    main()
