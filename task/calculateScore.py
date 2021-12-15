from builtins import str
from builtins import range
from psychopy import visual, event, core, gui, data, misc
from os.path import exists, join

import time, random, math, os
import numpy as np
import copy
import csv

from psychopy import prefs
prefs.general['audioLib'] = ['sounddevice']
from psychopy import sound

from psychopy import logging

path  = "/Users/normanlab/Documents/karina/adderzip_fMRI/exp_fMRI/"

#set up dialog box 
expInfo = {'participant ID':'000'}

#set variables for responses
PID = expInfo['participant ID']

#import scoreData
scoreData = open(path+'data/forcedChoiceData/7/clickData/clickData_'+PID+'_7.csv')
scoreData = csv.reader(scoreData)
scoreData = list(scoreData)
scoreData = scoreData[1::]
scoreData = np.array(scoreData)

score = np.mean(scoreData[:,9])
print(score)

