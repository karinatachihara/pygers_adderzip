'''
based on adderzipMT17
stand-alone version for after fMRI

created on 02/10/20
03/02/20: fixed error message for conditions 2-4 (expInfo type)
'''

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

path  = "/Users/normanlab/Documents/karina/adderzip_fMRI/exp_fMRI/"

#set up dialog box 
expInfo = {'Experimenter':'kt', 'condition':'1', 'participant ID':'000', 'run':13}

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
if exists(join(path+'data/itemtestData/logs', 'itemtest_log_{0:03d}.txt'.format(
                int(PID)))):
    print("Log file already exists for this subject!!!")

logging.setDefaultClock(expTime)
log = logging.LogFile(f=join(path+'data/itemtestData/logs', 'itemtest_log_{0:03d}.txt'.format(
                          int(PID))), level=logging.INFO,
                      filemode='w')
initial_message = ("Starting forced choice phase: "
                   "subject {0}, run {1}, condition {2}".format(
                    int(PID),int(runNum), int(condID)))
logging.exp(initial_message)

trigger = 'equal'

#import testInfo
testInfo = open(path+'info/itemtestInfo/itemtestInfo_'+PID+'.csv')
testInfo = csv.reader(testInfo)
testInfo = list(testInfo)
testInfo = testInfo[1::]
testInfo = np.array(testInfo)

fixTime = 0.5
prefixTime = 0.8

respWait = 5 #max time waiting for a response 

textdispTime = 0.75
fbTime = 0.75

ITI = 0.8

class itemTest_exp:
    def __init__(self):
        #self.win = visual.Window(fullscr=False) #TEMP
        self.win = visual.Window(size=[1920, 1080],fullscr=True)
        self.mouse = event.Mouse(visible=True,newPos=[0,0],win=self.win)
        self.trialClock = core.Clock()
        self.globalClock = core.Clock()
        self.clickClock = core.Clock()
        
        posR = [0.3,0.5]
        posL = [-0.3,0.5]
        posStart = [0,-0.8]
        
        self.buttonR = visual.Circle(self.win, radius=0.05,lineWidth = 0, fillColor=(0, 0, 5), fillColorSpace='rgb', pos=posR)
        self.buttonL = visual.Circle(self.win, radius=0.05,lineWidth = 0, fillColor=(0, 0, 5), fillColorSpace='rgb', pos=posL)
        self.buttonStart = visual.Circle(self.win, radius=0.05,lineWidth = 0, fillColor=(1,1,1), fillColorSpace='rgb', pos=posStart)
        
        #create texts
        self.messageHit = visual.TextStim(self.win, pos=[0,-0.4], wrapWidth = 50, text='Let me know when you are ready to start.')
        
        self.messageTest = visual.TextStim(self.win, pos=[0,0.4], wrapWidth = 50, text="Choose the image associated with the word you hear.")
        
        self.messageRecall = visual.TextStim(self.win, pos=[0,0], wrapWidth = 50, text="Press D for the left image and J for the right image.")
        
        self.messageSlow=visual.TextStim(self.win, pos=[0, 0], wrapWidth = 50, text='Please answer faster.')
                     
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


    def itemTest(self,testInfo):
        print("itemTest")

        self.messageTest.draw()
        self.messageRecall.draw()
        self.messageHit.draw()
        self.win.flip()
        event.waitKeys()

        trialNum = self.gr2ItemNum *3
        itemTestData = np.zeros((trialNum,5))

        itemTestData[:,0] = 8
        itemTestData[:,1] = np.arange(trialNum)

        #loop through each test
        for eachTest in range(trialNum):
            
            #draw fixation
            self.fixation.draw()
            self.win.flip()
            core.wait(fixTime)
            
            #play prefix
            preID = testInfo[eachTest,4]
            self.playPrefix(preID)
            
            #play partWord
            partWordID = str(int(testInfo[eachTest,5]))
            self.playPartWord(partWordID)
            
            #set up image 1
            imID1 = str(int(testInfo[eachTest,12]))
            imCat1 = testInfo[eachTest,10]
            imFile1 = self.setImage(imID1,imCat1)
            
            #set up image2
            imID2 = str(int(testInfo[eachTest,13]))
            imCat2 = testInfo[eachTest,11]
            imFile2 = self.setImage(imID2,imCat2)
            
            if testInfo[eachTest,14] == 1:
                pos1 = [-0.5,0] #correct on left
                pos2 = [0.5,0] #correct on right
            else:
                pos1 = [0.5,0] #correct on right
                pos2 = [-0.5,0] #correct on left
            
            image1 = visual.ImageStim(win=self.win, image = imFile1, mask=None, pos=pos1)
            image2 = visual.ImageStim(win=self.win, image = imFile2, mask=None, pos=pos2)

            scale = 0.5
            image1.size *= scale/max(image1.size)
            image2.size *= scale/max(image2.size)
            
            image1.draw()
            image2.draw()
            self.win.flip()
            testRt=0
            self.trialClock.reset()
            
            testResp=None 
            
            while testResp==None:
                allKeys=event.waitKeys(maxWait=respWait)
                
                if allKeys==None: 
                    self.messageSlow.draw()
                    self.win.flip()
                    core.wait(textdispTime)
                    testResp=-5
                    testRt = -5.000
                    break
                    
                testRt=self.trialClock.getTime()
                
                for thisKey in allKeys:
                    if thisKey=='d':
                        testResp = 1
                        
                    elif thisKey=='j':
                        testResp = 0
                        
                    elif thisKey in ['escape']:
                        core.quit() #abort experiment
                
                event.clearEvents() #must clear other (eg mouse) events - they clog the buffer
            
            testCorAns = testInfo[eachTest, 14]
            if testCorAns == testResp:
                testAcc = 1
            else:
                testAcc = 0
            
            itemTestData[eachTest,2] = testRt
            itemTestData[eachTest,3] = testResp
            itemTestData[eachTest,4] = testAcc
            

        np.savetxt(path+"data/itemtestData/itemtest_"+PID+".csv", itemTestData, fmt='%f', delimiter=",", header = 'blockNum,testItemNum, testRt, testResp, testAcc')

        return

    def setImage(self,imID, imCat):#function to set up image
        print('setImage')
        if imCat == "1":
            imFile = path + "stimuli/female/female" + imID + ".jpg"
            
        elif imCat == "0":
            imFile = path + "stimuli/indoor/indoor" + imID + ".jpg"
        
            
        return imFile

    def playPrefix(self,preID):#function to draw fixation and play prefix
        print('playPrefix')
        #play prefix 
        if preID == "1":
            preName = "abber"
        elif preID == "0":
            preName = "belling"
            
        prefix = sound.Sound(value = path+"stimuli/prefix_wav/"+preName+".wav", stereo = True)
        prefix.play()
        core.wait(prefixTime)
        return
    
    def playPartWord(self,partWordID):#function to play partWord
        print('playPartWord')
        #play partWord
        partWord = sound.Sound(value = path+"stimuli/partWords_wav/pw_"+partWordID+".wav", stereo = True)
        dur = partWord.getDuration()
        #print("duration=",dur)
        #print("time before=", trialClock.getTime())
        partWord.play()
        core.wait(dur)
        #print("time after=", trialClock.getTime())


exp = itemTest_exp()
exp.itemTest(testInfo)