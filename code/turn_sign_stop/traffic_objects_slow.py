from threading import Timer
import logging


class TrafficObject(object):

    def set_car_state(self, car_state):
        pass

    @staticmethod
    def is_close_by(obj, frame_height,obj_height, min_height_pct=0.05):
        # default: if a sign is 10% of the height of frame
        #obj_height = obj.bbox[1][1]-obj.bbox[0][1]
        obj_height = obj_height
        return obj_height / frame_height > min_height_pct

class keep(TrafficObject):
    pass


class RedTrafficLight(TrafficObject):

    def set_car_state(self, car_state):
        logging.debug('red light: stopping car')
        car_state['speed'] = 0


class GreenTrafficLight(TrafficObject):

    def set_car_state(self, car_state):
        logging.debug('green light: make no changes')


class Turn_sign(TrafficObject):
    def __init__(self, speed_limit):
        self.speed_limit = speed_limit

    def set_car_state(self, car_state):
        logging.debug('turn sign: set limit to %d' % self.speed_limit)
        car_state['speed_limit'] = self.speed_limit

class Person(TrafficObject):

    def set_car_state(self, car_state):
        logging.debug('pedestrian: stopping car')

        car_state['speed'] = 0


class SpeedLimit(TrafficObject):

    def __init__(self, speed_limit):
        self.speed_limit = speed_limit

    def set_car_state(self, car_state):
        logging.debug('speed limit: set limit to %d' % self.speed_limit)
        car_state['speed_limit'] = self.speed_limit


class StopSign(TrafficObject):
    """
    Stop Sign object would wait
    """

    def __init__(self, wait_time_in_sec=3, min_no_stop_sign=20):
        self.in_wait_mode = False   # 정지 기다리는지 여부
        self.has_stopped = False    # 이미 정지했는지 여부
        self.wait_time_in_sec = wait_time_in_sec    # 정지 시간
        self.min_no_stop_sign = min_no_stop_sign    # 정지 아닐 때 정지 유지 최소 프레임
        self.no_stop_count = min_no_stop_sign       # 정지 아닐 때 프레임 카운트
        self.timer = None

    def set_car_state(self, car_state):
        self.no_stop_count = self.min_no_stop_sign

        if self.in_wait_mode:   # 이미 정지한 경우
            logging.debug('stop sign: 2) still waiting')
            # wait for 2 second before proceeding
            car_state['speed'] = 0
            return

        if not self.has_stopped:    # 정지 아닌 경우
            logging.debug('stop sign: 1) just detected')

            car_state['speed'] = 0
            self.in_wait_mode = True    # 정지상태로 전환
            self.has_stopped = True
            self.timer = Timer(self.wait_time_in_sec, self.wait_done)
            self.timer.start()
            return

    def wait_done(self):        # 정지 신호 끝나면 호출
        logging.debug('stop sign: 3) finished waiting for %d seconds' % self.wait_time_in_sec)
        self.in_wait_mode = False

    def clear(self):
        if self.has_stopped:
            # need this counter in case object detection has a glitch that one frame does not
            # detect stop sign, make sure we see 20 consecutive no stop sign frames (about 1 sec)
            # and then mark has_stopped = False
            self.no_stop_count -= 1
            if self.no_stop_count == 0:
                logging.debug("stop sign: 4) no more stop sign detected")
                self.has_stopped = False
                self.in_wait_mode = False  # may not need to set this