#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
imagine phase in fMRI

10/02/19: change TR to 1.5 and volume to 4
10/10/19: get rid of launchScan. Check for key input instead. Take out HOVER from name. add logging
10/11: fixed vol counting
10/18: defrankenization
11/01: instructions
02/04/2020: added 7 TR in the beginning and end with blank screen
02/06: changed output file name convention to include run; got rid of posTrack for consistency
02/10: 5 TR in the beginning 10 TRs in the end; moved up face and scene button 
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

path  = "/Users/normanlab/Documents/karina/adderzip_fMRI/exp_fMRI/"

#set up dialog box 
expInfo = {'Experimenter':'kt', 'condition':'1', 'participant ID':'000', 'run':4}

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
if exists(join(path+'data/imagineData/'+str(runNum)+'/logs', 'imagine_log_{0:03d}.txt'.format(
                int(PID)))):
    print("Log file already exists for this subject!!!")

logging.setDefaultClock(expTime)
log = logging.LogFile(f=join(path+'data/imagineData/'+str(runNum)+'/logs', 'imagine_log_{0:03d}.txt'.format(
                          int(PID))), level=logging.INFO,
                      filemode='w')
initial_message = ("Starting imagine phase: "
                   "subject {0}, run {1}, condition {2}".format(
                    int(PID),int(runNum), int(condID)))
logging.exp(initial_message)

trigger = 'equal'

#import imagineInfo
imagineInfo = open(path+'info/imagineInfo/'+str(runNum)+'/imagineInfo_'+PID+'.csv')
imagineInfo = csv.reader(imagineInfo)
imagineInfo = list(imagineInfo)
imagineInfo = imagineInfo[1::]
imagineInfo = np.array(imagineInfo)

#import jitterInfo
jitterInfo = open(path+'info/jitterInfo/'+str(runNum)+'/jitterInfo_'+PID+'.csv')
jitterInfo = csv.reader(jitterInfo)
jitterInfo = list(jitterInfo)
jitterInfo = jitterInfo[1::]
jitterInfo = np.array(jitterInfo)

leadTR = 5
endTR = 10

class imagineRun:

    def __init__(self):
        #self.win = visual.Window(fullscr=False) #TEMP
        self.win = visual.Window(size=[1920, 1080],fullscr=True)
        #self.win = visual.Window(size=[1200, 800],fullscr=True)
        self.mouse = event.Mouse(visible=True,newPos=[0,0],win=self.win)
        self.trialClock = core.Clock()
        self.globalClock = core.Clock()

        #create texts
        self.messageHit = visual.TextStim(self.win, pos=[0,-0.4], wrapWidth = 50, text='Let me know when you are ready to start.')
        self.messageSlow=visual.TextStim(self.win, pos=[0, 0], wrapWidth = 50, text='Please answer faster.')
        self.messageImagine = visual.TextStim(self.win, pos = [0,0.4],wrapWidth = 50, text = 'Imagine, as vividly as possible, \nthe image associated with the word you hear.')
        self.messageHover = visual.TextStim(self.win, pos=[0,0], wrapWidth = 50, text="Then, hover over the word that describes the picture.")
        self.messageDone=visual.TextStim(self.win, pos=[0, 0], wrapWidth = 50, text='You are done with this run! \nPlease wait a few minutes while we set the next one up.')
         

        #create keys and buttons
        posLeftKey = [-0.3,-0.6]
        posRightKey = [0.3,-0.6]
        self.leftKey = visual.TextStim(self.win, pos = posLeftKey,wrapWidth = 50, text = 'Face')
        self.rightKey = visual.TextStim(self.win, pos = posRightKey,wrapWidth = 50, text = 'Scene')
        
        self.buttonLeftKey = visual.Circle(self.win, radius=0.5,lineWidth = 0, fillColor=(0, 0, 0), fillColorSpace='rgb', pos=posLeftKey)
        self.buttonRightKey = visual.Circle(self.win, radius=0.5,lineWidth = 0, fillColor=(0, 0, 0), fillColorSpace='rgb', pos=posRightKey)

        #create fixation
        self.fixation = visual.TextStim(self.win, text = '+', color=-1, colorSpace='rgb')

        #number of items in each group
        self.gr1ItemNum = 8
        self.gr2ItemNum = 8
        self.gr3ItemNum = 8
        self.gr4ItemNum = 24
        self.gr5ItemNum = 8
        self.gr6ItemNum = 24

        
        self.totalItemNum = self.gr1ItemNum + self.gr2ItemNum + self.gr3ItemNum + self.gr4ItemNum + self.gr5ItemNum
        self.totalItemNumExtra = self.gr1ItemNum + self.gr2ItemNum + self.gr3ItemNum + self.gr4ItemNum + self.gr5ItemNum + self.gr6ItemNum


    def imaginePhase(self,imagineInfo,jitterInfo):
        #print("imaginePhase")
        logging.info('imagine phase')

        trialNum,placeHolder = np.shape(imagineInfo)

        self.messageImagine.draw()
        self.messageHover.draw()
        self.messageHit.draw()
        self.win.flip()
        event.waitKeys()
        logging.info('instructions over')

        clickDataFile = open(path + 'data/imagineData/'+str(runNum)+'/clickData/clickData_'+PID+'_'+str(runNum)+'.csv', 'w')
        clickDataFile.write('trialCt,rt,response,accuracy,pos_x,pos_y\n')
        
        trackAllDataFile = open(path + 'data/imagineData/'+str(runNum)+'/trackAllData/trackAllData_'+PID+'_'+str(runNum)+'.csv', 'w')
        trackAllDataFile.write('trialCt,rt,pos_x,pos_y\n')

        timingDataFile = open(path + 'data/imagineData/'+str(runNum)+'/timingData/timingData_'+PID+'_'+str(runNum)+'.csv', 'w')
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

        for eachTrial in range(trialNum):
        #for eachTrial in range(3):

            trialTime = self.trialClock.reset()
            logging.info('trial begins. trialTime reset')

            self.mouse.clickReset()

            globalTime = self.globalClock.getTime()
            timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'g',globalTime))
            trialTime = self.trialClock.getTime()
            timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'t',trialTime))

            self.fixation.draw()
            self.leftKey.draw()
            self.rightKey.draw()
            self.win.flip()

            #play word
            preID = str(int(imagineInfo[eachTrial,5]))
            pwID = str(int(imagineInfo[eachTrial,2]))
            self.playWord(preID,pwID)

            duration = np.float(jitterInfo[eachTrial,5])
            trialTime = self.trialClock.getTime()

            #count TRs during trial
            while trialTime < duration:
                trialTime = self.trialClock.getTime()
                #print("while loop")
                theseKeys = event.getKeys()
                if 'escape' in theseKeys:
                    break
                if trigger in theseKeys:
                    #print("trigger!")
                    logging.info('Trigger received')

                    vol += 1

                    globalTime = self.globalClock.getTime()
                    timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'g',globalTime))
                    trialTime = self.trialClock.getTime()
                    timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'t',trialTime))

                movingPos = self.mouse.getPos()
                timeStamp = self.trialClock.getTime()
                
                trackAllDataFile.write('%i,%f,%f,%f\n'%(eachTrial,timeStamp,movingPos[0],movingPos[1]))
            
                resp,pos = self.checkClick()

                rt = self.trialClock.getTime()
                
                goal = int(imagineInfo[eachTrial,6])
                acc = self.compareBehaviorToGoal(resp,goal)

            globalTime = self.globalClock.getTime()
            timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'g',globalTime))
            trialTime = self.trialClock.getTime()
            timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'t',trialTime))       

            clickDataFile.write('%i,%f,%c,%i,%f,%f\n'%(eachTrial,rt,resp,acc,pos[0],pos[1]))
            
            event.clearEvents() #must clear other (eg mouse) events - they clog the buffer

            #wait for last TR
            #print("ready for last TR")
            logging.info('ready for last trigger')
            self.win.flip()
            wait = True
            while wait: # first scanner trigger
                theseKeys = event.getKeys()
                # check for quit:
                if "escape" in theseKeys:
                    self.win.close()
                    core.quit()
                if trigger in theseKeys: 
                    logging.info('Last trigger received')
                    vol+=1
                    wait = False
            wait = True

            globalTime = self.globalClock.getTime()
            timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'g',globalTime))
            trialTime = self.trialClock.getTime()
            timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'t',trialTime))

            logging.info('trial ends')

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
        timingDataFile.close()

        return 

    def playWord(self,preID,partWordID):
        #print('play word')
        logging.info('play word')

        if preID == "1":
            preName = "abber"
        elif preID== "0":
            preName = "belling"

        word = sound.Sound(value = path+"stimuli/partWords_wav/"+preName+"/"+preName+"Word"+partWordID+".wav", stereo = True)
        word.play()

        return


    def checkClick(self):
        pos = self.mouse.getPos()

        if self.buttonRightKey.contains(pos):
            resp = 'R'
        elif self.buttonLeftKey.contains(pos):
            resp =  'L'
        else:
            resp = 'B'
        
        return (resp,pos)

    def compareBehaviorToGoal(self,resp,goal):
        if resp == 'R':
            resp = 0
        elif resp == 'L':
            resp = 1
            
        if resp==goal:
            acc = 1
        elif resp == 'B':
            acc = -1
        else:
            acc = 0

        return(acc)


exp = imagineRun()
exp.imaginePhase(imagineInfo,jitterInfo)
