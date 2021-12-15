'''
based on adderzipMT_07_22_19
prep for fMRI version

02/21/20: This is a blank screen for testing 
'''

from psychopy import core, visual, gui, data, misc, event
import time, random, math, os
import numpy as np
import copy
import csv
from os.path import exists, join


from psychopy import prefs
prefs.general['audioLib'] = ['sounddevice']
from psychopy import sound

from psychopy import logging
logging.console.setLevel(logging.WARNING)

path  = "/Users/normanlab/Documents/karina/adderzip_fMRI/exp_fMRI/"

class testRun:
    def __init__(self):
        self.win = visual.Window(size=[1920, 1080],fullscr=True)
        #self.win = visual.Window(size=[800, 600],fullscr=False)
        self.mouse = event.Mouse(visible=True,newPos=[0,0],win=self.win)
        
        #create keys and buttons
        posLeftKey = [-1,0]
        posRightKey = [1,0]
        posTopKey = [0,1]
        posBottomKey = [0,-1]
        self.leftKey = visual.TextStim(self.win, pos = posLeftKey,wrapWidth = 50, text = 'Left')
        self.rightKey = visual.TextStim(self.win, pos = posRightKey,wrapWidth = 50, text = 'Right')
        self.topKey = visual.TextStim(self.win, pos = posLeftKey,wrapWidth = 50, text = 'Top')
        self.bottomKey = visual.TextStim(self.win, pos = posRightKey,wrapWidth = 50, text = 'Bottom')

    def test(self):
        
        self.leftKey.draw()
        self.rightKey.draw()
        self.topKey.draw()
        self.bottomKey.draw()
        self.win.flip()
        
        while True:
            
            keys = event.getKeys()
            if keys == ['escape']:
                core.quit()

exp = testRun()
exp.test()


