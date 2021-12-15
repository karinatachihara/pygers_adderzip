'''
based on adderzipMT_07_22_19
prep for fMRI version

It takes the all-in-one code and splits it by different phases/runs.
This one goes through the exposure phase.
This one hovers over face and scene (built for in scanner)

10/18: defrankenization - play word in one go, add logging. 
10/22: no ITI, longer fix time instead (more consistent with the rest)
11/01: instructions
02/04/2020: added 7 TR in the beginning and end with blank screen
02/10: 5 TR in the beginning 10 TRs in the end; moved up face and scene button
02/22: end of scan message 
03/02/20: fixed error message for conditions 2-4 (expInfo type)
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

#set up dialog box 
expInfo = {'Experimenter':'kt', 'condition':'1', 'participant ID':'000'}

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

#timing
fixTime = 1
respTime = 4.5

# Start PsychoPy's clock (mostly for logging)
expTime = core.Clock()
runNum = 1

# Set up PsychoPy's logging function
if exists(join(path+'data/exposureData/logs', 'exposure_log_{0:03d}.txt'.format(
                int(PID)))):
    print("Log file already exists for this subject!!!")

logging.setDefaultClock(expTime)
log = logging.LogFile(f=join(path+'data/exposureData/logs', 'exposure_log_{0:03d}.txt'.format(
                          int(PID))), level=logging.INFO,
                      filemode='w')
initial_message = ("Starting exposure phase: "
                   "subject {0}, run {1}, condition {2}".format(
                    int(PID),int(runNum), int(condID)))
logging.exp(initial_message)

#trigger for the scanner
trigger = 'equal'

#import exposusreInfo
expoInfo = open(path+'info/exposureInfo/exposureInfo_'+PID+'.csv')
expoInfo = csv.reader(expoInfo)
expoInfo = list(expoInfo)
expoInfo = expoInfo[1::]
expoInfo = np.array(expoInfo)

leadTR = 5
endTR = 10

class exposureRun:
    def __init__(self):
        self.win = visual.Window(size=[1920, 1080],fullscr=True)
        #self.win = visual.Window(size=[800, 600],fullscr=False)
        self.mouse = event.Mouse(visible=True,newPos=[0,0],win=self.win)
        self.trialClock = core.Clock()
        self.expClock = core.Clock()
      
        #create texts
        self.messageHit = visual.TextStim(self.win, pos=[0,-0.4], wrapWidth = 50, text='Let me know when you are ready to start.')
        
        self.messageExposure = visual.TextStim(self.win, pos=[0,0.4], wrapWidth = 50, text="Hover over the word that describes the picture \n(face or scene).")
                
        self.messageSlow=visual.TextStim(self.win, pos=[0, 0], wrapWidth = 50, text='Please answer faster.')

        self.messageDone=visual.TextStim(self.win, pos=[0, 0], wrapWidth = 50, text='You are done with this run! \nPlease wait a few minutes while we set the next one up.')
        
        #create keys and buttons
        posLeftKey = [-0.3,-0.6]
        posRightKey = [0.3,-0.6]
        self.leftKey = visual.TextStim(self.win, pos = posLeftKey,wrapWidth = 50, text = 'Face')
        self.rightKey = visual.TextStim(self.win, pos = posRightKey,wrapWidth = 50, text = 'Scene')
        
        self.buttonLeftKey = visual.Circle(self.win, radius=0.05,lineWidth = 0, fillColor=(0, 0, 0), fillColorSpace='rgb', pos=posLeftKey)
        self.buttonRightKey = visual.Circle(self.win, radius=0.05,lineWidth = 0, fillColor=(0, 0, 0), fillColorSpace='rgb', pos=posRightKey)
        
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


    def exposurePhase(self,expoInfo):
        #print('exposurePhase')
        logging.info('exposure phase')

        #display messages
        self.messageExposure.draw()
        self.messageHit.draw()
        self.win.flip()
        event.waitKeys()
        logging.info('instructions over')

        clickDataFile = open(path + 'data/exposureData/clickData/clickData_'+PID+'.csv', 'w')
        clickDataFile.write('trialCt,rt,response,accuracy,pos_x,pos_y\n')
        
        trackAllDataFile = open(path + 'data/exposureData/trackAllData/trackAllData_'+PID+'.csv', 'w')
        trackAllDataFile.write('trialCt,rt,pos_x,pos_y\n')

        eachTrial = -1
        vol = 0

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


        #for eachTrial in range(5):
        for eachTrial in range(self.totalItemNum):

            self.trialClock.reset()
            logging.info('trial begins. trialTime reset')
            self.mouse.clickReset()
            
            #draw fixation
            self.fixation.draw()
            self.win.flip()
            logging.info('fixation begins')
            core.wait(fixTime)

            #set up image
            imCat = int(expoInfo[eachTrial,5])
            imID = str(int(expoInfo[eachTrial,6]))
            imFile = self.setImage(imID,imCat)
            
            image = visual.ImageStim(win=self.win, image = imFile, mask=None, pos=[0,0])
            
            image.draw()
            self.fixation.draw()
            self.leftKey.draw()
            self.rightKey.draw()
            self.win.flip()
            logging.info('show image')
            
            #play word
            preID = str(int(expoInfo[eachTrial,4]))
            pwID = str(int(expoInfo[eachTrial,1]))
            self.playWord(preID,pwID)

            #record response
            acc = 0
            rt = 0
            pos = [0,0]
            
            acc, rt, pos, resp = self.collectClicks(eachTrial,expoInfo,trackAllDataFile)

            self.win.flip()
            logging.info('end of trial')
            
            clickDataFile.write('%i,%f,%c,%i,%f,%f\n'%(eachTrial,rt,resp,acc,pos[0],pos[1]))
            
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

        return

    def setImage(self,imID, imCat):#function to set up image
        logging.info('set image')
        #print('setImage')

        if imCat == 1:
            imFile = path + "stimuli/female/female" + imID + ".jpg"
            
        elif imCat == 0:
            imFile = path + "stimuli/indoor/indoor" + imID + ".jpg"
            
        return imFile

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

    def collectClicks(self,eachTrial,imagineInfo,trackAllDataFile):
        logging.info('collect clicks')
        #print("collectClicks")
        
        vol = 0
        while True:
            
            keys = event.getKeys()
            if keys == ['escape']:
                core.quit()

            if trigger in keys:
                #print("trigger!")
                logging.info('Trigger received')

                vol += 1
            movingPos, timeStamp = self.posTrack()
            
            trackAllDataFile.write('%i,%f,%f,%f\n'%(eachTrial,timeStamp,movingPos[0],movingPos[1]))
        
            resp,pos = self.checkClick()

            rt = self.trialClock.getTime()
            
            goal = int(expoInfo[eachTrial,5])
            acc = self.compareBehaviorToGoal(resp,goal)

            if timeStamp>respTime:
                break
                     
        return (acc,rt,pos,resp)

    def posTrack(self):
        movingPos = self.mouse.getPos()
        timeStamp = self.trialClock.getTime()
        return(movingPos,timeStamp)

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

exp = exposureRun()
exp.exposurePhase(expoInfo)
