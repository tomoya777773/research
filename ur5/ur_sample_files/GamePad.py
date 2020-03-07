import pygame
from pygame.locals import *

class GamePad(object):
    init_flag_ = False

    def __init__(self, game_pad_id):
        if not self.init_flag_:
            pygame.joystick.init()
            pygame.init()
            init_flag_ = True
        try:
            self.joys_ = pygame.joystick.Joystick(game_pad_id)
            self.joys_id_ = game_pad_id
            self.joys_.init()
            self.InitValue()
        except pygame.error:
            print('Not found gamepad')
    
    def InitValue(self):
        self.name_ = self.joys_.get_name()
        self.axes_ = [0] * self.joys_.get_numaxes()
        self.buttons_ = [False] * self.joys_.get_numbuttons()
        self.hats_ = [(0, 0)] * self.joys_.get_numhats()
        
        self.button_mode_ = [0] * self.joys_.get_numbuttons()
        # 0:down_ture_up_false 1:down_trigger 2:up_trigger 3:hold
        self.pre_buttons_ = [False] * self.joys_.get_numbuttons()

    def Update(self):
        for e in pygame.event.get():
            if e.type == JOYAXISMOTION and e.joy == self.joys_id_:
                self.axes_[e.axis] = e.value
            elif e.type == JOYHATMOTION and e.joy == self.joys_id_:
                self.hats_[e.hat] = e.value
            elif e.type == JOYBUTTONUP and e.joy == self.joys_id_:
                if self.button_mode_[e.button] == 0:
                    self.buttons_[e.button] = False
                elif self.button_mode_[e.button] == 2:
                    self.buttons_[e.button] = True
            elif e.type == JOYBUTTONDOWN and e.joy == self.joys_id_:
                if self.button_mode_[e.button] == 0:
                    self.buttons_[e.button] = True
                elif self.button_mode_[e.button] == 1:
                    self.buttons_[e.button] = True
                """
                elif self.button_mode_[e.button] == 3:
                    if self.buttons_[e.button] == False:
                        self.buttons_[e.button] = True
                    else:
                        self.buttons_[e.button] = False
                """
        for i in range(len(self.buttons_)):
            if self.button_mode_[i] == 1:
                if self.pre_buttons_[i]:
                    self.buttons_[i] = False
            if self.button_mode_[i] == 2:
                if self.pre_buttons_[i]:
                    self.buttons_[i] = False
        self.pre_buttons_[:] = self.buttons_[:]
        
    def GetAxis(self, axis_id):
        return self.axes_[axis_id]
        
    def GetButton(self, button_id):
        return self.buttons_[button_id]
    
    def GetHat(self, hat_id):
        return self.hats_[hat_id]
