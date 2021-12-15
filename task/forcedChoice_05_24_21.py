'''
based on adderzipMT_08_27_19 & starPilot11 (because it will be on the tablet)
prep for fMRI version

It takes the all-in-one code and splits it by different phases/runs.
This one goes through the learning blocks - forced choice trials.
This one takes which run into account and also waits for post-feedback movement if incorrect. Also fixed scrambled image bug. 

10/02: attempt to TR lock at the end of each trial, during normal ITI
10/08: fixing data collection issue (collect as you go)
10/11: get rid of launchScan. Check for key input instead. add logging
10/17: tested on P001. needed fix (sending timingdatafile to other functions)
10/18: fixed run and trial numer correspondence (begin & end) to account for 2 neural measures; defrankenization
10/22: feedback & min ITI timing; fixed rt recording 
11/01: instructions
11/08: fixed crash (crash when it was slow ish on click start, and timing issue created on 10/22 by setting trialTime) by creating clickTime
02/04/2020: added 7 TR in the beginning and end with blank screen
02/06: changed output file name convention to include run, fixed clickClock by getting rid of posTrack
02/10: 5 TR in the beginning 10 TRs in the end; instructions fit on the screen.
02/22: end of scan message
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
expInfo = {'Experimenter':'kt', 'condition':'1', 'participant ID':'000', 'run':2}

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
if exists(join(path+'data/forcedChoiceData/'+str(runNum)+'/logs', 'forcedChoice_log_{0:03d}.txt'.format(
                int(PID)))):
    print("Log file already exists for this subject!!!")

logging.setDefaultClock(expTime)
log = logging.LogFile(f=join(path+'data/forcedChoiceData/'+str(runNum)+'/logs', 'forcedChoice_log_{0:03d}.txt'.format(
                          int(PID))), level=logging.INFO,
                      filemode='w')
initial_message = ("Starting forced choice phase: "
                   "subject {0}, run {1}, condition {2}".format(
                    int(PID),int(runNum), int(condID)))
logging.exp(initial_message)

trigger = 'equal'

#import runInfo
runInfo = open(path+'info/runInfo/runInfo_'+PID+'.csv')
runInfo = csv.reader(runInfo)
runInfo = list(runInfo)
runInfo = runInfo[1::]
runInfo = np.array(runInfo)

#range for this run
if runNum == 2:
     begin = 0
     end = 56
     #end = 3 #TEMP
elif runNum == 3:
     begin = 56
     end = 232
elif runNum == 6:
     begin = 232
     end = 400
elif runNum == 7:
     begin = 400
     end = 576

time_feedback = 0.5
time_minITI = 0.5

leadTR = 5
endTR = 10

class forcedChoiceRun:
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
        
        self.messageRet = visual.TextStim(self.win, pos=[0,0.5], wrapWidth = 50, text="After you hear the whole word, \nhover over the white button.\nThen, move to the blue button under \nthe image that goes with that word.")
       
        self.messageWrong = visual.TextStim(self.win, pos=[0,0], wrapWidth = 50, text="If you are wrong, move to the blue button under the correct image.")
        
        self.messageSlow=visual.TextStim(self.win, pos=[0, 0], wrapWidth = 50, text='Please answer faster.')
             
        self.messageDone=visual.TextStim(self.win, pos=[0, 0], wrapWidth = 50, text='You are done with this run! \nPlease wait a few minutes while we set the next one up.')
                
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

    def forcedChoicePhase(self,runInfo):
        #print("forcedChoicePhase")
        logging.info('forced choice phase')

        self.messageRet.draw()
        self.messageHit.draw()
        self.messageWrong.draw()
        self.win.flip()
        event.waitKeys()
        logging.info('instructions over')

        clickDataFile = open(path + 'data/forcedChoiceData/'+str(runNum)+'/clickData/clickData_'+PID+'_'+str(runNum)+'.csv', 'w')
        clickDataFile.write('trialCt,rt1,accuracy,rt2,pos_x,pos_y,rt3,pos_x,pos_y,score\n')
        
        trackAllDataFile = open(path + 'data/forcedChoiceData/'+str(runNum)+'/trackAllData/trackAllData_'+PID+'_'+str(runNum)+'.csv', 'w')
        trackAllDataFile.write('trialCt,rt,pos_x,pos_y\n')
        
        trackDataFile = open(path + 'data/forcedChoiceData/'+str(runNum)+'/trackData/trackData_'+PID+'_'+str(runNum)+'.csv', 'w')
        trackDataFile.write('trialCt,rt,pos_x,pos_y\n')

        timingDataFile = open(path + 'data/forcedChoiceData/'+str(runNum)+'/timingData/timingData_'+PID+'_'+str(runNum)+'.csv', 'w')
        timingDataFile.write('trial,TR,clockType,clock\n')

        totalTrialNum,placeHolder = np.shape(runInfo)

        sumScore = 0

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

        for eachTrial in range(begin,end):

            trialTime = self.trialClock.reset()
            logging.info('trial begins. trialTime reset')

            globalTime = self.globalClock.getTime()
            timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'g',globalTime))
            trialTime = self.trialClock.getTime()
            timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'t',trialTime))


            self.buttonDisplay()
            
            #play word
            preID = str(int(runInfo[eachTrial,4]))
            pwID = str(int(runInfo[eachTrial,5]))
            duration  = self.playWord(preID,pwID)

            slow, rt1, vol = self.clickStart(duration, vol,eachTrial,timingDataFile)
            
            acc = 0
            rt2 = 0
            pos = [0,0]
            
            rt3 = 0
            FBPos = [0,0]

            trialType = int(runInfo[eachTrial,2])
            
            if not(slow):
                corrImage = self.showImages(eachTrial,runInfo)

                needCorr = 0
                acc, rt2, pos, posData, vol = self.collectClicks(eachTrial,runInfo,trackAllDataFile,trackDataFile,needCorr,vol,timingDataFile)

                maxDev = self.getMaxDev(eachTrial,posData)

                score = self.getScore(maxDev,rt2, acc)

                sumScore = sumScore + score

                vol = self.provideFeedback(trialType,corrImage,score,acc,vol,eachTrial,timingDataFile)
                logging.info('provide feedback')
                
                if (trialType < 6 and acc == 0):
                    needCorr = 1
                    na, rt3, FBPos, posData, vol = self.collectClicks(eachTrial,runInfo,trackAllDataFile,trackDataFile,needCorr,vol,timingDataFile)
            else:
                score = -1

            self.win.flip()

            globalTime = self.globalClock.getTime()
            timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'g',globalTime))
            trialTime = self.trialClock.getTime()
            timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'t',trialTime))         
            
            event.clearEvents() #must clear other (eg mouse) events - they clog the buffer

            #wait for last TR
            #print("ready for last TR")
            logging.info('ready for last trigger')
            self.win.flip()

            beforeWaiting = self.trialClock.getTime()

            wait = True
            while wait: # first scanner trigger
                timeCheck = self.trialClock.getTime()

                theseKeys = event.getKeys()
                # check for quit:
                if "escape" in theseKeys:
                    self.win.close()
                    core.quit()
                if trigger in theseKeys: 
                    logging.info('Last trigger received')
                    vol+=1

                    #even if it gets a trigger wait if it's been less than 0.5 after trial ends
                    if timeCheck - beforeWaiting > time_minITI:
                        logging.info('that was the last trigger. move on.')
                        wait = False
                    else:
                        logging.info('jk too soon for trigger. try again.')
            wait = True

            globalTime = self.globalClock.getTime()
            timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'g',globalTime))
            trialTime = self.trialClock.getTime()
            timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'t',trialTime))

            #print("ready for clickData")
            clickDataFile.write('%i,%f,%i,%f,%f,%f,%f,%f,%f,%i\n'%(eachTrial,rt1,acc,rt2,pos[0],pos[1],rt3,FBPos[0],FBPos[1],score))

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
        trackDataFile.close()
        timingDataFile.close()

        avgScore = sumScore / totalTrialNum
        print("avgScore = ",avgScore)
        #np.savetxt(path+"info/score.csv", avgScore, fmt='%i', delimiter=",", header = 'final score')
        
        return

    def buttonDisplay(self):
        self.buttonR.draw()
        self.buttonL.draw()
        self.buttonStart.draw()
        self.win.flip()
        return

    def playWord(self,preID,partWordID):
        #print('play word')
        logging.info('play word')

        if preID == "1":
            preName = "abber"
        elif preID== "0":
            preName = "belling"

        #print(preID,partWordID)
        word = sound.Sound(value = path+"stimuli/partWords_wav/"+preName+"/"+preName+"Word"+partWordID+".wav", stereo = True)
        duration = word.getDuration()
        word.play()
        logging.info('play word')

        return duration


    def clickStart(self, duration, vol,eachTrial,timingDataFile):

        pretrial = True
        slow = False
        logging.info('click start - listening to TR')

        while pretrial == True:
            
            press = self.mouse.getPressed()
            timeCheck = self.trialClock.getTime()

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

            if timeCheck > duration:
                if press[0]:
                    resp,pos = self.checkClick()
                    #print("resp",resp,"pos", pos)
                    rt = self.trialClock.getTime()
                    
                    event.clearEvents()
                    
                    if resp == 'S':
                        logging.info('clicked start button')
                        pretrial = False

            rt1Max = 4 + duration
            if timeCheck > rt1Max:
                logging.info('too slow for pretrial')
                rt = timeCheck
                self.tooSlow()
                slow=True
                pretrial=False
            
        return slow,rt, vol

    def tooSlow(self):
        tooSlow = visual.TextStim(self.win, text='Too slow')
        tooSlow.draw()
        self.win.flip()
        core.wait(0.5)
        return

    def showChoices(self,eachTrial,runInfo):
        trialType = runInfo[eachTrial,6]
        if trialType ==1:
            corrImage = self.showImages(runInfo,eachTrial)
        else:
            corrImage = self.showStar(runInfo,eachTrial)
        return (corrImage,trialType)

    def showImages(self,eachTrial,runInfo):
        #print ("showImages")

        imCat1 = int(runInfo[eachTrial,10])
        imCat2 = int(runInfo[eachTrial,11])

        imID1 = str(int(runInfo[eachTrial,12]))
        imID2 = str(int(runInfo[eachTrial,13]))
        
        imFile1 = self.getImFile(imCat1,imID1)
        imFile2 = self.getImFile(imCat2,imID2)
        
        whichLeft = int(runInfo[eachTrial,14])
        prefix = int(runInfo[eachTrial,4])
        
        posL = [-0.3,0.8]
        posR = [0.3,0.8]
        
        if whichLeft == 1:
            imPos1 = posL
            imPos2 = posR
        elif whichLeft == 0:
            imPos1 = posR
            imPos2 = posL
            
        scale = 0.5
        im1 = visual.ImageStim(win=self.win, image = imFile1, mask=None, pos=imPos1)
        im2 = visual.ImageStim(win=self.win, image = imFile2, mask=None, pos=imPos2)
        im1.size *= scale/max(im1.size)
        im2.size *= scale/max(im2.size)
        
        corrImage = im1
        
        im1.draw()
        im2.draw()
        self.buttonR.draw()
        self.buttonL.draw()
        self.buttonStart.draw()
        self.win.flip()
        logging.info('show images')
        
        return(corrImage)

    def getImFile(self,imCat,imID):
        print("getImFile")

        if imCat == 0:
            imFile = path + "stimuli/indoor/indoor" + imID + ".jpg"
        elif imCat == 1:
            imFile = path + "stimuli/female/female" + imID + ".jpg"
        elif imCat == 2:
            imFile = path + "stimuli/indoor_scrambled/indoor_scr" + imID + ".jpg"
        elif imCat == 3:
            imFile = path + "stimuli/female_scrambled/female_scr" + imID + ".jpg"

        return(imFile)

    def checkClick(self):
        pos = self.mouse.getPos()

        if self.buttonR.contains(pos):
            resp = 'R'
        elif self.buttonL.contains(pos):
            resp =  'L'
        elif self.buttonStart.contains(pos):
            resp = 'S'
        else:
            resp = 'B'
        
        return (resp,pos)

    def collectClicks(self,eachTrial,runInfo,trackAllDataFile,trackDataFile, needCorr, vol,timingDataFile):
        #print('collectClicks')
        logging.info('collect clicks')
        
        self.clickClock.reset()
        self.mouse.clickReset()

        lastMovingPos = [0,0]
        lastTimeStamp = 0

        posData = np.array([0,0])

        trialType = int(runInfo[eachTrial,2])
        
        while True:
            
            theseKeys = event.getKeys()
            if "escape" in theseKeys:
                self.win.close()
                core.quit()

            if trigger in theseKeys:
                #print("trigger!")
                logging.info('Trigger received')

                vol += 1

                globalTime = self.globalClock.getTime()
                timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'g',globalTime))
                trialTime = self.trialClock.getTime()
                timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'t',trialTime))
                clickTime = self.clickClock.getTime()
                timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'c',clickTime))
            
            thisTime = self.clickClock.getTime()
            
            rtMax = 3
            if thisTime > rtMax:
                logging.info('too slow to make a choice')
                rt = self.trialClock.getTime()
                acc = 0
                pos = self.mouse.getPos()
                posData = np.vstack((posData,pos))
                self.tooSlow()
                #print(thisTime, self.trialClock.getTime())
                break
            

            diff = thisTime - lastTimeStamp
            
            if diff>0.02:
                movingPos = self.mouse.getPos()
                timeStamp = self.clickClock.getTime()
                
                trackAllDataFile.write('%i,%f,%f,%f\n'%(eachTrial,timeStamp,movingPos[0],movingPos[1]))
                
                posData = np.vstack((posData,movingPos))

                if not(movingPos[0] == lastMovingPos[0] and movingPos[1] == lastMovingPos[1]):
                    trackDataFile.write('%i,%f,%f,%f\n'%(eachTrial,timeStamp,movingPos[0],movingPos[1]))
                 
                lastMovingPos = movingPos
                lastTimeStamp = timeStamp
            

            resp,pos = self.checkClick()

            if (resp=='R' or resp=='L'):
                logging.info('got R or L response')

                rt = self.trialClock.getTime()
                
                goal = int(runInfo[eachTrial,14])
                if trialType>5:
                    if goal == 0:
                        goal = 1
                    elif goal == 1:
                        goal = 0

                acc = self.compareBehaviorToGoal(resp,goal)

                if needCorr == 0:
                    break

                if needCorr == 1:
                    if acc == 1:
                        break
        #print("collect clicks posData",posData)
        return (acc,rt,pos, posData, vol)

    def outsideButton(self):
        outside = visual.TextStim(self.win, text='Make sure to click the buttons')
        outside.draw()
        self.win.flip()
        core.wait(0.5)
        return

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

    def provideFeedback(self,trialType, corrImage, score, acc,vol,eachTrial,timingDataFile):
        if trialType < 7:
            corrImage.draw()

        #if acc == 1:
        scoreText = visual.TextStim(self.win, pos=[0,0], wrapWidth = 50, text=str(score))
        #else:
        #    scoreText = visual.TextStim(self.win, pos=[0,0], wrapWidth = 50, text="0")

        scoreText.draw()

        self.buttonDisplay()

        beforeFBTime = self.trialClock.getTime()
        timeCheck = self.trialClock.getTime()

        while timeCheck - beforeFBTime < time_feedback:
            timeCheck = self.trialClock.getTime()

            theseKeys = event.getKeys()
            if "escape" in theseKeys:
                self.win.close()
                core.quit()

            if trigger in theseKeys:
                #print("trigger!")
                logging.info('Trigger received')

                vol += 1

                globalTime = self.globalClock.getTime()
                timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'g',globalTime))
                trialTime = self.trialClock.getTime()
                timingDataFile.write('%i,%f,%c,%f\n'%(eachTrial,vol,'t',trialTime))
        return vol

    def getScore(self, dev, rt, acc):
        print("getScore")

        devScore = 200 * dev

        rtScore = 10 * rt

        score = devScore + rtScore

        if score > 100:
            score = 99

        score = 100 - score
        #flip so higher score is better

        score = np.floor(score)
        score = int(score)

        if acc == 0:
            score = int(0)

        return (score)

    def getMaxDev(self,eachTrial,posData):
        print("getMaxDev")
        #print(posData)

        end_x = posData[-1,0]
        end_y = posData[-1,1]

        start_x = posData[1,0]
        start_y = posData[1,1]

        change_x = end_x - start_x
        change_y = end_y - start_y

        slope = change_y / change_x
        intercept = start_y - slope*start_x

        dataSize, placeHolder = posData.shape

        allDevs = []

        for eachPoint in range(dataSize):
            x = posData[eachPoint,0]
            y = posData[eachPoint,1]

            yStraight = slope*x + intercept

            yDiff = y - yStraight

            thisDev = np.array([yDiff * math.cos(math.atan(slope))])
            allDevs = np.hstack((allDevs,thisDev)) 

        maxDev = max(allDevs)

        if change_x == 0 or change_y == 0:
            maxDev = 0

        return (maxDev)




exp = forcedChoiceRun()
exp.forcedChoicePhase(runInfo)

