import cv2
import logging
import datetime
import time

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

from PIL import Image
from traffic_objects import *

_SHOW_IMAGE = False


class ObjectsOnRoadProcessor(object):
    """
    This class 1) detects what objects (namely traffic signs and people) are on the road
    and 2) controls the car navigation (speed/steering) accordingly
    """

    def __init__(self,
                 car=None,
                 speed_limit=40,
                 width=320,
                 height=240,
                 threshold =0.1):
        # model: This MUST be a tflite model that was specifically compiled for Edge TPU.
        # https://coral.withgoogle.com/web-compiler/
        logging.info('Creating a ObjectsOnRoadProcessor...')
        self.width = width
        self.height = height

        # initialize car
        self.car = car
        self.speed_limit = speed_limit
        self.speed = speed_limit
        self.threshold = 10

        model='/home/pi/AI-self-driving-RC-car/code/test/data/mobilenet_v2_haram.tflite'
        label='/home/pi/AI-self-driving-RC-car/code/test/data/labelmap_haram.txt'

        # initialize TensorFlow models
        # with open(label, 'r') as f:
        #     pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        #     self.labels = dict((int(k), v) for k, v in pairs)

        # initial edge TPU engine
        logging.info('Initialize Edge TPU with model %s...' % model)
        self.interpreter = make_interpreter(model)
        self.interpreter.allocate_tensors()
        self.labels = read_label_file(label)
        self.inference_size = input_size(self.interpreter)
        logging.info('Initialize Edge TPU with model done.')

        # initialize open cv for drawing boxes

        self.boxLineWidth = 1
        self.lineType = 2
        self.annotate_text = ""
        self.annotate_text_time = time.time()
        self.time_to_show_prediction = 1.0  # ms

        #
        self.traffic_objects = {0: GreenTrafficLight(),
                                1: Person(),
                                2: RedTrafficLight(),
                                3: SpeedLimit(25),
                                4: SpeedLimit(40),
                                5: StopSign()}

    def process_objects_on_road(self, frame):

        #objects, final_frame = self.detect_objects(frame)
        cv2_im = frame
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, self.inference_size)
        run_inference(self.interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(self.interpreter,self.threshold)[:10]    #number of categories with highest score to display
        cv2_im = self.append_objs_to_img(cv2_im, objs)

        #self.control_car(objs)

        return cv2_im

    def append_objs_to_img(self, cv2_im,  objs):
        height, width, _ = cv2_im.shape
        scale_x, scale_y = width / self.inference_size[0], height / self.inference_size[1]
        for obj in objs:
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)
    
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, self.labels.get(obj.id, obj.id))
    
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        return cv2_im

    def control_car(self, objects):
        logging.debug('Control car...')
        car_state = {"speed": self.speed_limit, "speed_limit": self.speed_limit}

        if len(objects) == 0:
            logging.debug('No objects detected, drive at speed limit of %s.' % self.speed_limit)

        contain_stop_sign = False
        for obj in objects:
            obj_label = self.labels[obj.label_id]
            processor = self.traffic_objects[obj.label_id]
            if processor.is_close_by(obj, self.height):
                processor.set_car_state(car_state)
            else:
                logging.debug("[%s] object detected, but it is too far, ignoring. " % obj_label)
            if obj_label == 'Stop':
                contain_stop_sign = True

        if not contain_stop_sign:
            self.traffic_objects[5].clear()

        self.resume_driving(car_state)

    def resume_driving(self, car_state):
        old_speed = self.speed
        self.speed_limit = car_state['speed_limit']
        self.speed = car_state['speed']

        if self.speed == 0:
            self.set_speed(0)
        else:
            self.set_speed(self.speed_limit)
        logging.debug('Current Speed = %d, New Speed = %d' % (old_speed, self.speed))

        if self.speed == 0:
            logging.debug('full stop for 1 seconds')
            time.sleep(1)

    def set_speed(self, speed):
        # Use this setter, so we can test this class without a car attached
        self.speed = speed
        if self.car is not None:
            logging.debug("Actually setting car speed to %d" % speed)
            self.car.back_wheels.speed = speed



    ############################
    # Frame processing steps
    ############################
    '''def detect_objects(self, frame):
        logging.debug('Detecting objects...')

        # call tpu for inference
        #start_time = time.time()

        cv2_im = frame
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, self.inference_size)
        run_inference(self.interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(self.interpreter,self.threshold)[:10]    #number of categories with highest score to display
        
        height, width, channels = cv2_im.shape
        scale_x, scale_y = width / self.inference_size[0], height / self.inference_size[1]
        for obj in objs:
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)
    
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, self.labels.get(obj.id, obj.id))
    
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                                self.font, 1.0, (255, 0, 0), 2)

        #cv2_im = self.append_objs_to_img(cv2_im, self.inference_size, objs, self.labels)

        # FPS
        # elapse_time = time.time() - start_time
        # fps = 1/elapse_time
        
        # cv2.putText(cv2_im, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.imshow('frame', cv2_im)

        return objs, cv2_im'''
    
#     def append_objs_to_img(cv2_im, inference_size, objs):
#         height, width, channels = cv2_im.shape
#         scale_x, scale_y = width / inference_size[0], height / inference_size[1]
#         for obj in objs:
#             bbox = obj.bbox.scale(scale_x, scale_y)
#             x0, y0 = int(bbox.xmin), int(bbox.ymin)
#             x1, y1 = int(bbox.xmax), int(bbox.ymax)
    
#             percent = int(100 * obj.score)
#             label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
    
#             cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
#             cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
#         return cv2_im


# ############################
# Utility Functions
############################
def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)


############################
# Test Functions
############################
def test_photo(file):
    object_processor = ObjectsOnRoadProcessor()
    frame = cv2.imread(file)
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Detected Objects', combo_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_stop_sign():
    # this simulates a car at stop sign
    object_processor = ObjectsOnRoadProcessor()
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/stop_sign.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Stop 1', combo_image)
    time.sleep(1)
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/stop_sign.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Stop 2', combo_image)
    time.sleep(2)
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/stop_sign.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Stop 3', combo_image)
    time.sleep(1)
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/green_light.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Stop 4', combo_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_video(video_file):
    object_processor = ObjectsOnRoadProcessor()
    cap = cv2.VideoCapture(video_file)

    # video_type = cv2.VideoWriter_fourcc(*'XVID')

    # video_overlay = cv2.VideoWriter("%s_overlay_%s.avi" % (video_file, date_str), video_type, 20.0, (320, 240))
    
    
    while cap.isOpened():
        start_time = time.time()
        try:
            _, frame = cap.read()

            combo_image = object_processor.process_objects_on_road(frame)

            # FPS
            elapse_time = time.time() - start_time
            fps = 1/elapse_time
            
            cv2.putText(combo_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('frame', combo_image)

            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                # video_overlay.release()
                cv2.destroyAllWindows()
                break
        except:
            print('error')
            break



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s:%(asctime)s: %(message)s')

    # These processors contains no state
    # test_photo('/home/pi/DeepPiCar/driver/data/objects/red_light.jpg')
    # test_photo('/home/pi/DeepPiCar/driver/data/objects/person.jpg')
    # test_photo('/home/pi/DeepPiCar/driver/data/objects/limit_40.jpg')
    # test_photo('/home/pi/DeepPiCar/driver/data/objects/limit_25.jpg')
    # test_photo('/home/pi/DeepPiCar/driver/data/objects/green_light.jpg')
    # test_photo('/home/pi/DeepPiCar/driver/data/objects/no_obj.jpg')
    test_video('/home/pi/AI-self-driving-RC-car/code/test/data/tmp/object2.avi')

    # test stop sign, which carries state
    #test_stop_sign()