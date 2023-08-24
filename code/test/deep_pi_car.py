import logging
import picar
import cv2
import datetime
import time
import PCA9685
import threading
from hand_coded_lane_follower_230820 import HandCodedLaneFollower


_SHOW_IMAGE = True
_SAVE_VIDEO = False

class DeepPiCar(object):

    __INITIAL_SPEED = 0
    __SCREEN_WIDTH = 320
    __SCREEN_HEIGHT = 240

    def __init__(self):
        """ Init camera and wheels"""
        logging.info('Creating a DeepPiCar...')

        # picar.setup()
        pwm=PCA9685.PWM(bus_number=1)
        pwm.setup()
        pwm.frequency = 60

        # set up camera
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(3, self.__SCREEN_WIDTH)
        self.camera.set(4, self.__SCREEN_HEIGHT)

        self.pan_servo = picar.Servo.Servo(1)
        self.pan_servo.offset = -30  # calibrate servo to center
        self.pan_servo.write(90)

        self.tilt_servo = picar.Servo.Servo(2)
        self.tilt_servo.offset = 20  # calibrate servo to center
        self.tilt_servo.write(90)

        logging.debug('Set up back wheels')
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.back_wheels.speed = 0  # Speed Range is 0 (stop) - 100 (fastest)

        logging.debug('Set up front wheels')
        self.front_wheels = picar.front_wheel_curve_test.Front_wheels()
        self.front_wheels.turning_offset = 125  # calibrate servo to center
        self.front_wheels.turn(90)  # Steering Range is 45 (left) - 90 (center) - 135 (right)

        self.back_wheels.speed = 

        # 주행 알고리즘 / 객체인식 수행
        self.lane_follower = HandCodedLaneFollower(self)
        # self.traffic_sign_processor = ObjectsOnRoadProcessor(self)
        # lane_follower = DeepLearningLaneFollower()

        # 비디오 저장
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        datestr = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.video_orig = self.create_video_recorder('../data/tmp/car_video%s.avi' % datestr)
        self.video_lane = self.create_video_recorder('../data/tmp/car_video_lane%s.avi' % datestr)
        self.video_objs = self.create_video_recorder('../data/tmp/car_video_objs%s.avi' % datestr)

        logging.info('Created a DeepPiCar')

    def create_video_recorder(self, path):
        return cv2.VideoWriter(path, self.fourcc, 20.0, (self.__SCREEN_WIDTH, self.__SCREEN_HEIGHT))

    def __enter__(self):
        """ Entering a with statement """
        return self

    def __exit__(self, _type, value, traceback):
        """ Exit a with statement"""
        if traceback is not None:
            # Exception occurred:
            logging.error('Exiting with statement with exception %s' % traceback)

        self.cleanup()

    def cleanup(self):
        """ Reset the hardware"""
        logging.info('Stopping the car, resetting hardware.')
        self.back_wheels.speed = 0
        self.front_wheels.turn(90)
        self.camera.release()
        self.video_orig.release()
        self.video_lane.release()
        self.video_objs.release()
        cv2.destroyAllWindows()


    def drive(self, speed=__INITIAL_SPEED):

        logging.info('Starting to drive at speed %s...' % speed)
        self.back_wheels.speed = speed
        while self.camera.isOpened():
            start_time = time.time()
            try:
                _, image_lane = self.camera.read()
                image_objs = image_lane.copy()
                
                # image_objs = self.process_objects_on_road(image_objs)
                # show_image('Detected Objects', image_objs)

                # 주행
                image_lane = self.follow_lane(image_lane)

                # FPS
                elapse_time = time.time() - start_time
                fps = 1/elapse_time
                
                cv2.putText(image_lane, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                show_image('Lane Lines', image_lane)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.cleanup()
                    break

                if _SAVE_VIDEO:
                    self.video_orig.write(image_lane)
                    self.video_lane.write(image_lane)
                    self.video_objs.write(image_objs)
            
            except:
                print('fail')


    # def process_objects_on_road(self, image):
    #     image = self.traffic_sign_processor.process_objects_on_road(image)
    #     return image

    def follow_lane(self, image):
        image = self.lane_follower.follow_lane(image)
        return image
    
# 멀티 스레딩
def image_processing_thread(car):
    
    while car.camera.isOpened():
        start_time = time.time()
        try:
            _, image_lane = car.camera.read()
            image_objs = image_lane.copy()
            
            # image_objs = self.process_objects_on_road(image_objs)
            # show_image('Detected Objects', image_objs)

            # 주행
            image_lane = car.follow_lane(image_lane)

            # FPS
            elapse_time = time.time() - start_time
            fps = 1/elapse_time
            
            cv2.putText(image_lane, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            show_image('Lane Lines', image_lane)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                car.cleanup()
                break

            if _SAVE_VIDEO:
                car.video_orig.write(image_lane)
                car.video_lane.write(image_lane)
                car.video_objs.write(image_objs)
        
        except:
            print('fail')

############################
# Utility Functions
############################
def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)


def main():
    with DeepPiCar() as car:
            image_thread = threading.Thread(target=image_processing_thread, args=(car,))
            image_thread.start()
            car.drive(40)
            image_thread.join()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s:%(asctime)s: %(message)s')
    
    main()