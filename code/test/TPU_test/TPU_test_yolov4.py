import time
import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

import utils as utils
from yolov4 import filter_boxes
# from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import time
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# from imutils.video import FPS


from pycoral.utils.edgetpu import make_interpreter


def main(_argv):
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    iou = 0.45
    score = 0.25


    input_size = 46
    video_path = '/home/pi/AI-self-driving-RC-car/code/test/data/tmp/object2.avi'


    vid = cv2.VideoCapture(video_path)

    interpreter = make_interpreter('/home/pi/AI-self-driving-RC-car/code/test/data/yolov3_int8.tflite')
    #interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

    #     # by default VideoCapture returns float instead of int
    # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(vid.get(cv2.CAP_PROP_FPS))
    # codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    # out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_id = 0
    while vid.isOpened():
        start_time = time.time()
        try:
            return_value, frame = vid.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

            
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            prev_time = time.time()


            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                #     boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                #                                     input_shape=tf.constant([input_size, input_size]))
                # else:
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                            input_shape=tf.constant([input_size, input_size]))


            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=iou,
                score_threshold=score
            )
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            image = utils.draw_bbox(frame, pred_bbox)
            curr_time = time.time()
            exec_time = curr_time - prev_time
            result = np.asarray(image)
            info = "time: %.2f ms" %(1000*exec_time)


            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # FPS
            elapse_time = time.time() - start_time
            fps = 1/elapse_time
            
            cv2.putText(result, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('frame', result)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                vid.release()
                cv2.destroyAllWindows()
                break

        except:
            print("eorror")
            break





if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass