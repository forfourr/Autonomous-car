import Servo
import filedb
import time



class Front_wheels(object):
    # front_wheel servo = 0
    CHANNEL = 0
    
    def __init__(self, db='config',bus_number=1, channel=CHANNEL):
        self.db = filedb.fileDB(db=db)
        self._turning_offset = int(self.db.get('turning_offset', default_value=0))

        self.angle_list = {
            'Left':0,
            'Right':250,
            'Straight':125,
            'little_right':150,
            'little_left':30
        }

        self.wheel = Servo.Servo(0, bus_number=1, offset=self.turning_offset)


    def turn_straight(self):
        self.wheel.write(self.angle_list['Straight'])

    def turn_left(self):
        self.wheel.write(self.angle_list['Left'])

    def turn_right(self):
        self.wheel.write(self.angle_list['Right'])
    
    def little_right(self):
        self.wheel.write(self.angle_list['little_right'])

    def turn(self, angle):
        if angle < self.angle_list['Left']:
            angle = self.angle_list['Left']
        if angle > self.angle_list['Right']:
            angle = self.angle_list['Right']
        self.wheel.write(angle)

    @property
    def turning_offset(self):
        return self._turning_offset
    
    @turning_offset.setter
    def turning_offset(self, value):
        if not isinstance(value, int):
            raise TypeError('"turning_offset" must be "int"')
        self._turning_offset = value
        self.db.set('turning_offset', value)
        self.wheel.offset = value
        self.turn_straight()
                

if __name__ == '__main__':
    
    # test
    Front_wheels = Front_wheels(channel=0)
    
    print("Left")
    Front_wheels.turn_left()
    time.sleep(2)

    print("Right")
    Front_wheels.turn_right()
    time.sleep(2)
    
    print('Straight')
    Front_wheels.turn_straight()
    time.sleep(2)

    print('test')
    Front_wheels.little_right()
    time.sleep(2)