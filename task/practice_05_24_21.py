'''
Practice phase outside of the scanner to get used to the timing of trials.
Praactice phase based on adderzipMT14

02/20/20: adapted for pre-fMRI practice (hover with mouse)
'''
from psychopy import core, visual, gui, data, misc, event
import time, random, math, os
import numpy as np
import copy

from psychopy import prefs
prefs.general['audioLib'] = ['sounddevice']
from psychopy import sound

from psychopy import logging
logging.console.setLevel(logging.WARNING)

path  = "/Users/normanlab/Documents/karina/adderzip_fMRI/"

fixTime = 0.5
prefixTime = 0.8

respWait = 5 #max time waiting for a response 

textdispTime = 0.75
fbTime = 0.75

ITI = 0.8

#set up dialog box 
expInfo = {'Experimenter':'kt', 'condition':1, 'participant ID':'000'}

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
condID = int(condID) 
condIDstr = str(condID)
PID = expInfo['participant ID']

class Practice:
    def __init__(self):
        self.win = visual.Window(size=[1200, 800],fullscr=True)
        self.mouse = event.Mouse(visible=True,newPos=[0,0],win=self.win)
        self.trialClock = core.Clock()
        self.expClock = core.Clock()
        
        posR = [0.3,0.5]
        posL = [-0.3,0.5]
        posStart = [0,-0.8]
        
        self.buttonR = visual.Circle(self.win, radius=0.05,lineWidth = 0, fillColor=(0, 0, 5), fillColorSpace='rgb', pos=posR)
        self.buttonL = visual.Circle(self.win, radius=0.05,lineWidth = 0, fillColor=(0, 0, 5), fillColorSpace='rgb', pos=posL)
        self.buttonStart = visual.Circle(self.win, radius=0.05,lineWidth = 0, fillColor=(1,1,1), fillColorSpace='rgb', pos=posStart)
        
        #create fixation
        self.fixation = visual.TextStim(self.win, text = '+', color=-1, colorSpace='rgb')
        
    def practiceTrials(self):
        print("practiceTrials")
        clickDataFile = open(path + 'data/practiceData/clickData/clickData_'+PID+'.csv', 'w')
        clickDataFile.write('trialCt,rt1,accuracy,rt2,pos_x,pos_y,rt3,pos_x,pos_y,score\n')
        
        trackAllDataFile = open(path + 'data/practiceData/trackAllData/trackAllData_'+PID+'.csv', 'w')
        trackAllDataFile.write('trialCt,rt,pos_x,pos_y\n')
        
        trackDataFile = open(path + 'data/practiceData/trackData/trackData_'+PID+'.csv', 'w')
        trackDataFile.write('trialCt,rt,pos_x,pos_y\n')
        
        animal1 = "bear"
        animal2 = "zebra"
        animalWord = "bear"
        left = 1

        self.animalTrial(animal1,animal2,animalWord,left, clickDataFile, trackAllDataFile, trackDataFile)

        animal1 = "cat"
        animal2 = "penguin"
        animalWord = "cat"
        left = 0

        self.animalTrial(animal1,animal2,animalWord,left, clickDataFile, trackAllDataFile, trackDataFile)

        animal1 = "chicken"
        animal2 = "lion"
        animalWord = "chicken"
        left = 0

        self.animalTrial(animal1,animal2,animalWord,left, clickDataFile, trackAllDataFile, trackDataFile)

        animal1 = "dog"
        animal2 = "cow"
        animalWord = "dog"
        left = 1

        self.animalTrial(animal1,animal2,animalWord,left, clickDataFile, trackAllDataFile, trackDataFile)

        animal1 = "elephant"
        animal2 = "koala"
        animalWord = "elephant"
        left = 0

        self.animalTrial(animal1,animal2,animalWord,left, clickDataFile, trackAllDataFile, trackDataFile) 

        animal1 = "mouse"
        animal2 = "monkey"
        animalWord = "mouse"
        left = 1

        self.animalTrial(animal1,animal2,animalWord,left, clickDataFile, trackAllDataFile, trackDataFile)
        
        clickDataFile.close()
        trackAllDataFile.close()
        trackDataFile.close()
        
        return

    def animalTrial(self,animal1,animal2,animalWord,left, clickDataFile, trackAllDataFile, trackDataFile):
        print("animalTrial")
        

        
        self.buttonDisplay()

        animalWord = sound.Sound(value = path+"stimuli/animalWords/animal_"+animalWord+".wav", stereo = True)
        animalWord.play() 
        core.wait(1)

        slow,rt1 = self.clickStart()
        
        rt3=0
        FBPos = [0,0]

        if not(slow):
            core.wait(0.2) #needs to wait to refresh

            posL = [-0.3,0.8]
            posR = [0.3,0.8]
            
            if left == 1:
                imPos1 = posL
                imPos2 = posR
            else:
                imPos1 = posR
                imPos2 = posL

            im1 = visual.ImageStim(win=self.win, image = path+"stimuli/animals/"+animal1+".jpg", mask=None, pos=imPos1)
            im2 = visual.ImageStim(win=self.win, image = path+"stimuli/animals/"+animal2+".jpg", mask=None, pos=imPos2)
            scale = 0.5
            im1.size *= scale/max(im1.size)
            im2.size *= scale/max(im2.size)

            im1.draw()
            im2.draw()
            self.buttonDisplay()

            corrImage = im1
            
            acc, rt2, pos,posData = self.practiceCollectClicks(left,animal1,trackAllDataFile,trackDataFile)

            dev = self.getMaxDev(1, posData)
            score = self.getScore(dev,rt2)
   
            self.provideFeedback(0,corrImage, score, acc)
            
            if acc == 0:
                na, rt3, FBPos, posData = self.practiceCollectClicks(left,animal1,trackAllDataFile,trackDataFile)

            clickDataFile.write('%s,%f,%i,%f,%f,%f,%f,%f,%f,%i\n'%(animal1,rt1,acc,rt2,pos[0],pos[1],rt3,FBPos[0],FBPos[1],score))
            
            if acc == 0:
                na = self.practiceCollectClicks(left,animal1,trackAllDataFile,trackDataFile)
        
        self.win.flip()
        core.wait(ITI)

        return
     
    def buttonDisplay(self):
        self.buttonR.draw()
        self.buttonL.draw()
        self.buttonStart.draw()
        self.win.flip()
        return
        
    def clickStart(self):
        self.trialClock.reset()
        pretrial = True
        slow = False
        while pretrial == True:
            #press = self.mouse.getPressed()
            timeCheck = self.trialClock.getTime()
            
            rt1Max = 3
            if timeCheck > rt1Max:
                rt = timeCheck
                self.tooSlow()
                slow=True
                pretrial=False
            
            #if press[0]:
            resp,pos = self.checkClick()
            rt = self.trialClock.getTime()
            
            event.clearEvents()
            
            if resp == 'S':
                pretrial = False
                rt = self.trialClock.getTime()
                    
        return slow,rt

    def practiceCollectClicks(self,left,animal1,trackAllDataFile,trackDataFile):
        print("practiceCollectClicks")

        self.trialClock.reset()
        self.mouse.clickReset()
        
        lastMovingPos = [0,0]
        lastTimeStamp = 0

        posData = np.array([0,0])
        
        while True:
            
            keys = event.getKeys()
            if keys == ['escape']:
                core.quit()
            
            thisTime = self.trialClock.getTime()
            
            rtMax = 3
            if thisTime > rtMax:
                rt = thisTime
                acc = 0
                pos = self.mouse.getPos()
                self.tooSlow()
                break
            
            diff = thisTime - lastTimeStamp
            
            if diff>0.02:
                movingPos, timeStamp = self.posTrack()
                
                trackAllDataFile.write('%s,%f,%f,%f\n'%(animal1,timeStamp,movingPos[0],movingPos[1]))
                
                posData = np.vstack((posData,movingPos))
               
                if not(movingPos[0] == lastMovingPos[0] and movingPos[1] == lastMovingPos[1]):
                    trackDataFile.write('%s,%f,%f,%f\n'%(animal1,timeStamp,movingPos[0],movingPos[1]))
               
                lastMovingPos = movingPos
                lastTimeStamp = timeStamp
                
            resp,pos = self.checkClick()

            if (resp=='R' or resp=='L'):
                print("got resp")

                rt = self.trialClock.getTime()
                
                goal = left

                acc = self.compareBehaviorToGoal(resp,goal)

                break
                 
        
        return(acc,rt, pos,posData)


    def tooSlow(self):
        tooSlow = visual.TextStim(self.win, text='Please answer faster')
        tooSlow.draw()
        self.win.flip()
        core.wait(textdispTime)
        return

    def posTrack(self):
        movingPos = self.mouse.getPos()
        timeStamp = self.trialClock.getTime()
        return(movingPos,timeStamp)

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

    def outsideButton(self):
        outside = visual.TextStim(self.win, text='Make sure to click the buttons')
        outside.draw()
        self.win.flip()
        core.wait(textdispTime)
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

    def provideFeedback(self,trialType, corrImage, score, acc):
        if trialType < 7:
        	corrImage.draw()

        print("acc=", acc)
        if acc == 1:
            print("passed")
            scoreText = visual.TextStim(self.win, pos=[0,0], wrapWidth = 50, text=str(score))
        else:
            scoreText = visual.TextStim(self.win, pos=[0,0], wrapWidth = 50, text="0")

        #scoreText = visual.TextStim(self.win, pos=[0,0], wrapWidth = 50, text=str(score))
        scoreText.draw()

        self.buttonDisplay()
        core.wait(fbTime)
        return

    def getMaxDev(self,curTrial,posData):
        print("getMaxDev")

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

    def getScore(self, dev, rt):
        print("getScore")

        devScore = 200 * dev
        print("devScore", devScore)

        rtScore = 10 * rt
        print("rtScore", rtScore)

        score = devScore + rtScore

        if score > 100:
            score = 99

        score = 100 - score
        #flip so higher score is better

        print(score)
        score = np.floor(score)
        score = int(score)

        return (score)

exp = Practice()
exp.practiceTrials()