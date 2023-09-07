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




class ObjectsOnRoadProcessor(object):
    """
    This class 1) detects what objects (namely traffic signs and people) are on the road
    and 2) controls the car navigation (speed/steering) accordingly
    """

    def __init__(self,
                 car=None,
                 speed_limit=40,
                 top_k=10, threshold=0.1,
                 video_path=None):
        self.top_k = top_k
        self.threshold = threshold
        self.video_path = video_path

        # initialize car
        self.car = car
        self.speed_limit = speed_limit
        self.speed = speed_limit
        self.height = 240
        self.width = 320
        
        model='/home/pi/AI-self-driving-RC-car/code/test/data/mobilenet_v2_haram.tflite'
        label_path='/home/pi/AI-self-driving-RC-car/code/test/data/labelmap_haram.txt'

        self.interpreter, self.inference_size = self.make_interpreter(model)
        self.labels = self.read_label_file(label_path)
        
        # self.traffic_objects = {0: GreenTrafficLight(),
        #                         1: Person(),
        #                         2: RedTrafficLight(),
        #                         3: SpeedLimit(25),
        #                         4: SpeedLimit(40),
        #                         5: StopSign()}
        
        self.traffic_objects = {0: keep(),          # Traffic Light
                                1: SpeedLimit(15),  # limit sign
                                2: StopSign(),      # stop sign
                                3: StopSign(),      # animal
                                4: StopSign(),      # car
                                5: StopSign(),      # human
                                6: Turn_right(),    # right_sign
                                7: Turn_left()}     # left_sign
        '''
        Traffic Light
        limit sign
        Stop sign
        animal
        car
        human
        right_sign
        left_sign
        EOF                
        '''

    def make_interpreter(self,model):
        interpreter = make_interpreter(model)
        interpreter.allocate_tensors()
        size = input_size(interpreter)
        return interpreter, size
    
    def read_label_file(self,label_path):
        labels = read_label_file(label_path)
        return labels



    def process_objects_on_road(self, frame):

        #objects, final_frame = self.detect_objects(frame)
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, self.inference_size)
        
        run_inference(self.interpreter, cv2_im_rgb.tobytes())
        #self.run_inference(cv2_im_rgb.tobytes())
        objs = get_objects(self.interpreter, self.threshold)[:self.top_k]

        print("Detected Objects:")
        for obj in objs:
            print(f"ID: {obj.id}, Score: {obj.score}, Bounding Box: {obj.bbox}")
            
        
        cv2_im = self.append_objs_to_img(cv2_im, objs)

        # 제어
        steer = self.control_car(objs)

        return cv2_im, steer

    def append_objs_to_img(self,cv2_im, objs):
        height, width, channels = cv2_im.shape
        self.scale_x, self.scale_y = width / self.inference_size[0], height / self.inference_size[1]
        for obj in objs:
            bbox = obj.bbox.scale(self.scale_x, self.scale_y)
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
        car_state = {"speed": self.speed_limit,
                     "speed_limit": self.speed_limit,
                     "steer":False}

        if len(objects) == 0:
            logging.debug('No objects detected, drive at speed limit of %s.' % self.speed_limit)

        
        contain_stop_sign = False
        for obj in objects:
            obj_label = self.labels[obj.id]

            bbox = obj.bbox.scale(self.scale_x, self.scale_y)
            y0 = int(bbox.ymin)
            y1 = int(bbox.ymax)
            obj_height= y1 - y0
            ## obj Object(id=6, score=0.22265625, bbox=BBox(xmin=62, ymin=0, xmax=287, ymax=28)) <class 'pycoral.adapters.detect.Object'>
            print(f"obj.id: {obj.id}")
            processor = self.traffic_objects[obj.id]
            if processor.is_close_by(obj, self.height, obj_height): #True/False
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
        self.speed_limit = car_state['speed_limit']     #제한속도
        self.speed = car_state['speed']     # 초기속도

        if self.speed == 0:
            print("STOOPPPP!!!!!")
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
# Test Functions
############################
def test_photo(file):
    object_processor = ObjectsOnRoadProcessor()
    frame = cv2.imread(file)
    combo_image = object_processor.process_objects_on_road(frame)
    cv2.imshow('Detected Objects', combo_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_stop_sign():
    # this simulates a car at stop sign
    object_processor = ObjectsOnRoadProcessor()
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/stop_sign.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    cv2.imshow('Stop 1', combo_image)
    time.sleep(1)
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/stop_sign.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    cv2.imshow('Stop 2', combo_image)
    time.sleep(2)
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/stop_sign.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    cv2.imshow('Stop 3', combo_image)
    time.sleep(1)
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/green_light.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    cv2.imshow('Stop 4', combo_image)

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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                # video_overlay.release()
                cv2.destroyAllWindows()
                break
        except Exception as e:
                print('Exception:', e)
                print('fail')
                break


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s:%(asctime)s: %(message)s')

    # These processors contains no state
    # test_photo('/home/pi/DeepPiCar/driver/data/objects/red_light.jpg')
    # test_photo('/home/pi/DeepPiCar/driver/data/objects/person.jpg')
    # test_photo('/home/pi/DeepPiCar/driver/data/objects/limit_40.jpg')
    # test_photo('/home/pi/DeepPiCar/driver/data/objects/limit_25.jpg')
    # test_photo('/home/pi/DeepPiCar/driver/data/objects/green_light.jpg')
    # test_photo('/home/pi/AI-self-driving-RC-car/code/test/data/tmp/obj_test.png')
    # test_video('/home/pi/AI-self-driving-RC-car/code/test/data/tmp/object2.avi')

    # test stop sign, which carries state
    #test_stop_sign()
    obj = ObjectsOnRoadProcessor()
    obj.set_speed(40)