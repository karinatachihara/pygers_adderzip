#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
localizer - copied from localiser_09_18_19 which was based on Paula's localizer

10/18/19: created
10/21/19: updated to work with self-made locInfo
10/22/19: fix timing - trials too short and ITI too long
11/01/19: return to button box response with shorter timing
02/04/2020: added 7 TR in the beginning and end with blank screen; make mouse invisible
02/10: 5 TR in the beginning 10 TRs in the end; edit instructions
02/22: end of scan message
03/02/20: fixed error message for conditions 2-4 (expInfo type)
"""

from __future__ import absolute_import, division, print_function

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
logging.console.setLevel(logging.WARNING)
#logging.console.setLevel(logging.INFO)

path  = "/Users/normanlab/Documents/karina/adderzip_fMRI/exp_fMRI/"

#set up dialog box 
expInfo = {'Experimenter':'kt', 'condition':'1', 'participant ID':'000', 'run':10}

#add the current time
expInfo['dateStr']= data.getDateStr() 
dlg = gui.DlgFromDict(expInfo, title='adderzip study', fixed=['dateStr'])

#the user hit cancel so exit
if dlg.OK:
   ready = 'start'
else: 
   core.quit()

#set variables for responses
condID = expInfo['condition'] 
PID = expInfo['participant ID']
runNum = expInfo['run']


# Start PsychoPy's clock (mostly for logging)
expTime = core.Clock()

# Set up PsychoPy's logging function
if exists(join(path+'data/localizerData/'+str(runNum)+'/logs', 'localizer_log_{0:03d}.txt'.format(
				int(PID)))):
	print("Log file already exists for this subject!!!")

logging.setDefaultClock(expTime)
log = logging.LogFile(f=join(path+'data/localizerData/'+str(runNum)+'/logs', 'localizer_log_{0:03d}.txt'.format(
						  int(PID))), level=logging.INFO,
					  filemode='w')
initial_message = ("Starting localizer phase: "
				   "subject {0}, run {1}, condition {2}".format(
					int(PID),int(runNum), int(condID)))
logging.exp(initial_message)

#fMRI responses
trigger = 'equal'
ansKeys = ['1']

#timing
time_showImage = 0.5
time_ITI = 1
time_lead = 0.1

#import info
locInfo = open(path+'info/locInfo/locInfo_'+PID+'.csv')
locInfo = csv.reader(locInfo)
locInfo = list(locInfo)
locInfo = locInfo[1::]
locInfo = np.array(locInfo)

#range for this run
if runNum == 10:
	begin = 0
	end = 90
elif runNum == 11:
	begin = 90
	end = 180
elif runNum == 12:
	begin = 180
	end = 270

leadTR = 5
endTR = 10

class localizerRun:

	def __init__(self):
		#self.win = visual.Window(fullscr=False) #TEMP
		self.win = visual.Window(size=[1920, 1080],fullscr=True)
		self.mouse = event.Mouse(visible=False,newPos=[0,0],win=self.win)
		self.trialClock = core.Clock()
		self.globalClock = core.Clock()

		#create texts
		self.messageHit = visual.TextStim(self.win, pos=[0,-0.4], wrapWidth = 50, text='Let me know when you are ready to start.')
		self.messageRepeat = visual.TextStim(self.win, pos = [0,0],wrapWidth = 50, text = 'Fixate on the cross in the the center of the screen.\nIf you see an image repeat, press the left button.')
		self.messageDone=visual.TextStim(self.win, pos=[0, 0], wrapWidth = 50, text='You are done with this run! \nPlease wait a few minutes while we set the next one up.')
         

		#create keys and buttons
		posLeftKey = [-0.3,-0.8]
		posRightKey = [0.3,-0.8]
		self.leftKey = visual.TextStim(self.win, pos = posLeftKey,wrapWidth = 50, text = 'OLD')
		self.rightKey = visual.TextStim(self.win, pos = posRightKey,wrapWidth = 50, text = 'NEW')
		
		self.buttonLeftKey = visual.Circle(self.win, radius=0.5,lineWidth = 0, fillColor=(0, 0, 0), fillColorSpace='rgb', pos=posLeftKey)
		self.buttonRightKey = visual.Circle(self.win, radius=0.5,lineWidth = 0, fillColor=(0, 0, 0), fillColorSpace='rgb', pos=posRightKey)

		#create fixation
		self.fixation = visual.TextStim(self.win, text = '+', color=-1, colorSpace='rgb')
		self.respFix = visual.TextStim(self.win, text = '+', color=1, colorSpace='rgb')

	def localizerPhase(self,locInfo):
		#print("imaginePhase")
		logging.info('imagine phase')

		trialNum,placeHolder = np.shape(locInfo)

		self.messageRepeat.draw()
		self.messageHit.draw()
		self.win.flip()
		event.waitKeys()
		logging.info('instructions over')

		clickDataFile = open(path + 'data/localizerData/'+str(runNum)+'/clickData/clickData_'+PID+'.csv', 'w')
		clickDataFile.write('trial,rt1,resp1,rt2,resp2,acc\n')
		
		trackAllDataFile = open(path + 'data/localizerData/'+str(runNum)+'/trackAllData/trackAllData_'+PID+'.csv', 'w')
		trackAllDataFile.write('trialCt,rt,pos_x,pos_y\n')

		timingDataFile = open(path + 'data/localizerData/'+str(runNum)+'/timingData/timingData_'+PID+'.csv', 'w')
		timingDataFile.write('trial,TR,clockType,clock\n')

		eachTrial = -1
		vol = 0

		globalTime = self.globalClock.getTime()
		timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'g',globalTime))
		trialTime = self.trialClock.getTime()
		timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'t',trialTime))

		#wait for first TR
		self.win.flip()
		wait = True
		while wait: 
			theseKeys = event.getKeys()
			# check for quit:
			if "escape" in theseKeys:
				self.win.close()
				core.quit()
			if trigger in theseKeys: 
				if vol == 0:
					logging.info('First trigger received')
				else: 
					logging.info('Trigger received')
				vol+=1
				if vol == leadTR:
					wait = False
		wait = True

		globalTime = self.globalClock.getTime()
		timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'g',globalTime))
		trialTime = self.trialClock.getTime()
		timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'t',trialTime))

		run_start = self.globalClock.getTime()

		for eachTrial in range(begin,end):
		#for eachTrial in range(3):

			trialTime = self.trialClock.reset()
			logging.info('trial begins. trialTime reset')

			#self.mouse.clickReset()

			rt1 = 1.5
			resp1 = 0
			rt2 = 1.5
			resp2 = 0

			trialOnset = float(locInfo[eachTrial,6])
			trialOffset = float(locInfo[eachTrial,7])

			#set up image
			imCat = int(locInfo[eachTrial,3])
			imID = str(int(locInfo[eachTrial,5]))

			if imCat == 0:
				imFile = path + 'stimuli/localizerStim/femaleFaces/female_' + imID + '.jpg'
			elif imCat == 1:
				imFile = path + 'stimuli/localizerStim/indoorScenes/indoor_' + imID + '.jpg'
			elif imCat == 2:
				imFile = path + 'stimuli/localizerStim/objects/manmade_' + imID + '.jpg'
			
			image = visual.ImageStim(win=self.win, image = imFile, mask=None, pos=[0,0])

			# check for escape
			theseKeys = event.getKeys()
			if "escape" in theseKeys:
				self.win.close()
				core.quit()

			globalTime = self.globalClock.getTime()
			trialStart = trialOnset + run_start
			#print(globalTime,trialStart)

			# wait to continue to code until relatively close to target fMRI trigger
			while (trialStart - globalTime) > time_lead:
				globalTime = self.globalClock.getTime()
				trialTime = self.trialClock.getTime()

				self.fixation.draw()
				self.win.flip()
				# check for quit:
				theseKeys = event.getKeys()
				if "escape" in theseKeys:
					self.win.close()
					core.quit()

				if len(np.intersect1d(theseKeys,ansKeys)) > 0:
						self.respFix.draw()
						self.win.flip()
						resp1 = int(theseKeys[-1])
						rt1 = trialTime

			logging.info('waiting til lead over')
			# check for escape
			theseKeys = event.getKeys()
			if "escape" in theseKeys:
				self.win.close()
				core.quit()

			wait = True
			while wait: 
				trialTime = self.trialClock.getTime()

				theseKeys = event.getKeys()

				# check for quit:
				if "escape" in theseKeys:
					self.win.close()
					core.quit()

				if len(np.intersect1d(theseKeys,ansKeys)) > 0:
					self.respFix.draw()
					self.win.flip()
					resp1 = int(theseKeys[-1])
					rt1 = trialTime

				if trigger in theseKeys: 
					logging.info('trigger received')
					vol+=1
					wait = False

			# check for escape
			theseKeys = event.getKeys()
			if "escape" in theseKeys:
				self.win.close()
				core.quit()

			image.draw()
			self.fixation.draw()
			#self.leftKey.draw()
			#self.rightKey.draw()
			self.win.flip()
			logging.info('image draw')

			globalTime = self.globalClock.getTime()
			timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'g',globalTime))
			trialTime = self.trialClock.getTime()
			timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'t',trialTime))

			actualOnset = trialTime
			#actualOnset = globalTime

			#self.mouse.clickReset()

			resp = 0

			while (trialTime - actualOnset) < time_showImage:
			#while (globalTime - actualOnset) < time_showImage:
				trialTime = self.trialClock.getTime()
				#globalTime = self.globalClock.getTime()

				theseKeys = event.getKeys()
				if "escape" in theseKeys:
					self.win.close()
					core.quit()

				if len(np.intersect1d(theseKeys,ansKeys)) > 0:
						self.respFix.draw()
						self.win.flip()
						resp2 = int(theseKeys[-1])
						rt2 = trialTime - actualOnset


				# movingPos = self.mouse.getPos()
				# timeStamp = time.time()
				
				# trackAllDataFile.write('%i,%f,%f,%f\n'%(eachTrial,timeStamp,movingPos[0],movingPos[1]))
			
				# pos = self.mouse.getPos()

				# if self.buttonRightKey.contains(pos):
				# 	if not(resp =='R'):
				# 		resp = 'R'
				# 		rt = trialTime
				# 		logging.info('Resp = R')
					
				# elif self.buttonLeftKey.contains(pos):
				# 	if not(resp =='L'):
				# 		resp =  'L'
				# 		rt = trialTime
				# 		logging.info('Resp = L')
					
				# else:
				# 	resp = 'B'
				# 	rt = trialTime
			
			logging.info('trial done')

			# self.fixation.draw()
			# self.win.flip()

			# if resp == 'R':
			# 	resp = 0
			# elif resp == 'L':
			# 	resp = 1
			# elif resp == 'B':
			# 	resp = -1

			goal = int(locInfo[eachTrial,4])
				
			# if resp==goal:
			# 	acc = 1
			# elif resp == 'B':
			# 	acc = -1
			# else:
			# 	acc = 0

			# print(resp,rt)

			if resp2==goal:
				acc = 1
			else:
				acc = 0

			#clickDataFile.write('%i,%f,%i,%i,%f,%f\n'%(eachTrial,rt,resp,acc,pos[0],pos[1]))
			clickDataFile.write('%i,%f,%i,%f,%i,%i\n'%(eachTrial,rt1,resp1,rt2,resp2,acc))
			event.clearEvents() #must clear other (eg mouse) events - they clog the buffer

		#wait for last TRs
		self.messageDone.draw()
		self.win.flip()
		wait = True
		lastVols = 0
		while wait: 
			theseKeys = event.getKeys()
			# check for quit:
			if "escape" in theseKeys:
				self.win.close()
				core.quit()
			if trigger in theseKeys: 
				logging.info('Trigger received - post trials')
				vol+=1
				lastVols+=1
				if lastVols == endTR:
					wait = False
		wait = True

		globalTime = self.globalClock.getTime()
		timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'g',globalTime))
		trialTime = self.trialClock.getTime()
		timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'t',trialTime))

		logging.info('run ends')


		clickDataFile.close()
		trackAllDataFile.close()


exp = localizerRun()
exp.localizerPhase(locInfo)
























