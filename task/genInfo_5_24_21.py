'''
based on adderzipMT_08_27_19
prep for fMRI version

It takes the all-in-one code and splits it by different phases/runs.
This one generates all the info files.
This one creates 2 imagine infos (1 for each run)

10/10/19: change TR and volume
10/21/19: added makeLocInfo
02/10/20: added itemtestInfo

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

path  = "/Users/normanlab/Documents/karina/adderzip_fMRI/exp_fMRI/"

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

#matrix that tells you the different groups and what their conditions are
condNum = 4
condDef = np.zeros((condNum,2))

#adder vs belling  for gr1
condDef[0:2,0]=1

#face vs scene for gr1
threeNums = np.arange(0,2)
condDef[0:2,1]=threeNums
condDef[2:4,1]=threeNums

condRow = condID - 1

np.savetxt(path+"info/condDef/condDef_"+PID+".csv", condDef, fmt='%i', delimiter=",", header = 'rule,face')

#matrix with participant info
partiInfo = np.array([expInfo['participant ID'],expInfo['condition'],expInfo['dateStr']])
partiInfo = partiInfo.reshape(1,partiInfo.shape[0])

np.savetxt(path+"info/partiInfo/partiInfo_"+PID+".csv", partiInfo, fmt='%s', delimiter=",", header = 'PID,condition,dateStr')


class gen_info:
    def __init__(self):
        self.win = visual.Window(size=[1200, 800],fullscr=True)
        self.mouse = event.Mouse(visible=True,newPos=[0,0],win=self.win)
        self.trialClock = core.Clock()
        self.expClock = core.Clock()

        #number of items in each group
        self.gr1ItemNum = 8
        self.gr2ItemNum = 8
        self.gr3ItemNum = 8
        self.gr4ItemNum = 24
        self.gr5ItemNum = 8
        self.gr6ItemNum = 24

        
        self.totalItemNum = int(self.gr1ItemNum + self.gr2ItemNum + self.gr3ItemNum + self.gr4ItemNum + self.gr5ItemNum)
        self.totalItemNumExtra = int(self.gr1ItemNum + self.gr2ItemNum + self.gr3ItemNum + self.gr4ItemNum + self.gr5ItemNum + self.gr6ItemNum)
        


    def makeMatchInfo(self,condRow,condDef):
        print('makeMatchInfo')
        
        matchInfo = np.zeros((self.totalItemNumExtra,7))
        
        #matchInfoNum
        matchInfo[:,0] = np.arange(self.totalItemNumExtra)
        
        #partWordID
        pwOrder = np.arange(self.totalItemNum)
        np.random.shuffle(pwOrder)
        matchInfo[:self.totalItemNum,1] = pwOrder
        
        #word group type
        ones = np.ones(self.gr1ItemNum)
        twos = np.ones(self.gr2ItemNum) * 2
        threes = np.ones(self.gr3ItemNum) * 3
        fours = np.ones(self.gr4ItemNum) * 4
        fives = np.ones(self.gr5ItemNum) * 5
        sixes = np.ones(self.gr6ItemNum) * 6
        groupInfo = np.hstack((ones,twos,threes,fours,fives,sixes))
        matchInfo[:,2] = groupInfo

        #count within the group
        ctWithin1 = np.arange(self.gr1ItemNum)
        ctWithin2 = np.arange(self.gr2ItemNum)
        ctWithin3 = np.arange(self.gr3ItemNum)
        ctWithin4 = np.arange(self.gr4ItemNum)
        ctWithin5 = np.arange(self.gr5ItemNum)
        ctWithin6 = np.arange(self.gr6ItemNum)
        ctWithin = np.hstack((ctWithin1,ctWithin2,ctWithin3,ctWithin4,ctWithin5,ctWithin6))
        matchInfo[:,3] = ctWithin
        
        #prefix type. adder or belling
        prefixNum_1 = self.gr1ItemNum + self.gr2ItemNum 
        prefixNum_2 = self.gr3ItemNum + self.gr4ItemNum + self.gr5ItemNum 
        prefixNum_0 = self.gr6ItemNum 

        if condDef[condRow,0] == 1:
        	prefix_1 = np.zeros(prefixNum_1)
        	prefix_2 = np.ones(prefixNum_2)

        else:
        	prefix_1 = np.ones(prefixNum_1)
        	prefix_2 = np.zeros(prefixNum_2)

        prefix_0 = np.zeros(prefixNum_0)
            
        prefixInfo= np.hstack((prefix_1,prefix_2,prefix_0))
        matchInfo[:,4] = prefixInfo
        
        #imCat. face or scene
        imCatNum_1 = self.gr1ItemNum 
        imCatNum_2 = self.gr2ItemNum
        imCatNum_3 = self.gr3ItemNum + self.gr4ItemNum
        imCatNum_4 = self.gr5ItemNum + self.gr6ItemNum

        if condDef[condRow,1] == 0:
            imCat_1 = np.zeros(imCatNum_1)
            imCat_2 = np.ones(imCatNum_2)
            imCat_3 = np.zeros(imCatNum_3)
            imCat_4 = np.ones(imCatNum_4)
        else: 
            imCat_1 = np.ones(imCatNum_1)
            imCat_2 = np.zeros(imCatNum_2)
            imCat_3 = np.ones(imCatNum_3)
            imCat_4 = np.zeros(imCatNum_4)
            
        imCatInfo = np.hstack((imCat_1,imCat_2,imCat_3,imCat_4))
        matchInfo[:,5] = imCatInfo
        
        #imID
        imID_1 = np.arange(1,imCatNum_1+1)
        imID_2 = np.arange(1,imCatNum_2+1)
        imID_3 = np.arange(imCatNum_1+1,(imCatNum_1+imCatNum_3+1))
        imID_4 = np.arange(imCatNum_2+1,(imCatNum_2+imCatNum_4+1))

        imIDInfo = np.hstack((imID_1,imID_2,imID_3,imID_4))
        matchInfo[:,6] = imIDInfo

        matchInfoExtra = copy.deepcopy(matchInfo)
        matchInfo = matchInfo[0:self.totalItemNum,:]
        
        np.savetxt(path+"info/matchInfoExtra/matchInfoExtra_"+PID+".csv", matchInfoExtra, fmt='%i', delimiter=",", header = 'matchInfoNum, partWordID, group, countWithin, prefix, imCat,imID')
        np.savetxt(path+"info/matchInfo/matchInfo_"+PID+".csv", matchInfo, fmt='%i', delimiter=",", header = 'matchInfoNum, partWordID, group, countWithin, prefix, imCat,imID')
        
        return matchInfo, matchInfoExtra

    def makeExpoInfo(self,matchInfo):
        print("make exposureInfo")

        #randomize order of each trial in trialDet
        trialDet = self.shuffleTrialDet(matchInfo, self.totalItemNum,0)

        exposureInfo = np.zeros((self.totalItemNum,9))
        exposureInfo[:,0:7] = trialDet
        exposureInfo[:,7] = np.arange((self.totalItemNum))
        exposureInfo[:,8] = 0

        np.savetxt(path+"info/exposureInfo/exposureInfo_"+PID+".csv", exposureInfo, fmt='%i', delimiter=",", header = 'matchInfoNum, partWordID, group, countWithin, prefix, imCat, imID, trialCt, eachBlock')

        return exposureInfo

    def shuffleTrialDet(self,matchInfo, trialPerBlock, eachBlock):#function to shuffle matchinfo 
        print('shuffleTrialDet')
        
        trialDet = copy.deepcopy(matchInfo) 
        
        if eachBlock==0:
            last = 0
        else:
            last = trialDet[-1,0]
            
        first = copy.deepcopy(last)
        
        while last==first:
            np.random.shuffle(trialDet)
            first = trialDet[0,0]
        
        return trialDet

    def makeRunInfo(self,matchInfoExtra):
        print("makeRunInfo")
        
        blockStruct = np.array([0,0,1,0,0,0,0,1])

        blockNum = len(blockStruct)

        grIDs_1 = np.arange(self.gr1ItemNum)
        grIDs_2 = np.arange(self.gr2ItemNum)
        grIDs_2 = grIDs_2 + self.gr1ItemNum
        grIDs_3 = np.arange(self.gr3ItemNum)
        grIDs_3 = grIDs_3 + (self.gr1ItemNum + self.gr2ItemNum)
        grIDs_4 = np.arange(self.gr4ItemNum)
        grIDs_4 = grIDs_4 + (self.gr1ItemNum + self.gr2ItemNum + self.gr3ItemNum)
        grIDs_5 = np.arange(self.gr5ItemNum)
        grIDs_5 = grIDs_5 + (self.gr1ItemNum + self.gr2ItemNum + self.gr3ItemNum + self.gr4ItemNum)
        grIDs_6 = np.arange(self.gr6ItemNum)
        grIDs_6 = grIDs_6 + (self.gr1ItemNum + self.gr2ItemNum + self.gr3ItemNum + self.gr4ItemNum + self.gr5ItemNum)

        for eachBlock in range(blockNum):

            star = blockStruct[eachBlock]

            if eachBlock == 0:
                runInfo = self.makeBlockInfo(eachBlock,star,matchInfoExtra,grIDs_1,grIDs_2,grIDs_3,grIDs_4,grIDs_5,grIDs_6)
            else:
                runInfo = np.vstack((runInfo, self.makeBlockInfo(eachBlock,star,matchInfoExtra,grIDs_1,grIDs_2,grIDs_3,grIDs_4,grIDs_5,grIDs_6)))

        totalTrialNum,placeHolder = np.shape(runInfo)
        runInfo[:,1] = np.arange(totalTrialNum)
        #add trial number 

        #print(runInfo)
        np.savetxt(path+"info/runInfo/runInfo_"+PID+".csv", runInfo, fmt='%i', delimiter=",", header = 'blockNum,trialNum,trialType,matchInfoNumAud,prefix,pwID,matchInfoNum1,matchInfoNum2,group type1,group type 2,imCat1,imCat2,imID1,imID2,whichLeft')
        
        return runInfo

    def makeBlockInfo(self,eachBlock,star,matchInfoExtra,grIDs_1,grIDs_2,grIDs_3,grIDs_4,grIDs_5,grIDs_6):
        print('makeBlockInfo')

        #No Scramble
        TT1 = self.makeChoices(grIDs_1, grIDs_1, grIDs_2,matchInfoExtra)
        TT2 = self.makeChoices(grIDs_2, grIDs_2, grIDs_1,matchInfoExtra)
        TT3 = self.makeChoices(grIDs_3, grIDs_3, grIDs_5,matchInfoExtra)
        TT4 = self.makeChoices(grIDs_4, grIDs_4, grIDs_6,matchInfoExtra)
        TT5 = self.makeChoices(grIDs_5, grIDs_5, grIDs_3,matchInfoExtra)


        #Scramble
        scramble = np.ones(8) * 200
        TT6 = self.makeChoices(grIDs_1, grIDs_1, scramble, matchInfoExtra)
        TT7 = self.makeChoices(grIDs_1, grIDs_2, scramble, matchInfoExtra)
        TT8 = self.makeChoices(grIDs_2, grIDs_2, scramble, matchInfoExtra)
        TT9 = self.makeChoices(grIDs_2, grIDs_1, scramble, matchInfoExtra)
        TT10 = self.makeChoices(grIDs_3, grIDs_3, scramble, matchInfoExtra)
        TT11 = self.makeChoices(grIDs_3, grIDs_5, scramble, matchInfoExtra)
        TT12 = self.makeChoices(grIDs_5, grIDs_5, scramble, matchInfoExtra)
        TT13 = self.makeChoices(grIDs_5, grIDs_3, scramble, matchInfoExtra)

        blockInfo = np.zeros((self.totalItemNum,15)) 

        if star == 0:
            blockInfo = self.makeNoStarBlock(eachBlock,star,matchInfoExtra,TT1,TT2,TT3,TT4,TT5)
        else:
            blockInfo = self.makeStarBlock(eachBlock,star,matchInfoExtra,TT1,TT2,TT3,TT4,TT5,TT6,TT7,TT8,TT9,TT10,TT11,TT12,TT13)

        return blockInfo

    def makeChoices(self,aud,cho1,cho2,matchInfoExtra):
        print('makeChoices')
        #print(matchInfo)


        np.random.shuffle(cho2)
        choices = np.vstack((aud,cho1))
        choices = np.vstack((choices,cho2))

        choices = np.transpose(choices)
        #print(choices)

        trialNum,placeHolder = np.shape(choices)

        choiceInfo = self.getDetails(trialNum,choices,matchInfoExtra)

        #print(choiceInfo)
        return choiceInfo

    def getDetails(self,trialNum,choices,matchInfoExtra):
        print("getDetails")
        choiceInfo = np.zeros((trialNum,11))

        for eachTrial in range(trialNum):
            matchInfoNumAud = int(choices[eachTrial,0])
            matchInfoNum1 = int(choices[eachTrial,1])
            matchInfoNum2 = int(choices[eachTrial,2])

            #print("hey",matchInfoNumAud)
            prefix = matchInfoExtra[matchInfoNumAud,4]
            pwID = matchInfoExtra[matchInfoNumAud,1]

            groupType1 = matchInfoExtra[matchInfoNum1,2]
            imCat1 = matchInfoExtra[matchInfoNum1,5]
            imID1 = matchInfoExtra[matchInfoNum1,6]

            if matchInfoNum2 == 200:
                #scrambled images
                groupType2 = 200
                imCat2 = imCat1 + 2
                imID2 = imID1

            else:
                groupType2 = matchInfoExtra[matchInfoNum2,2]
                imCat2 = matchInfoExtra[matchInfoNum2,5]
                imID2 = matchInfoExtra[matchInfoNum2,6]

            choiceInfo[eachTrial,0] = matchInfoNumAud
            choiceInfo[eachTrial,1] = prefix
            choiceInfo[eachTrial,2] = pwID
            choiceInfo[eachTrial,3] = matchInfoNum1
            choiceInfo[eachTrial,4] = matchInfoNum2
            choiceInfo[eachTrial,5] = groupType1
            choiceInfo[eachTrial,6] = groupType2
            choiceInfo[eachTrial,7] = imCat1
            choiceInfo[eachTrial,8] = imCat2
            choiceInfo[eachTrial,9] = imID1
            choiceInfo[eachTrial,10] = imID2


        return choiceInfo

    def makeNoStarBlock(self,eachBlock,star,matchInfoExtra,TT1,TT2,TT3,TT4,TT5):
        print('makeBlockInfo')

        trialNum = self.totalItemNum
        blockInfo = np.zeros((trialNum,15)) 

        #blockNum
        blockInfo[:,0] = eachBlock
        #trialNum
        blockInfo[:,1] = 0
        #fix this after all blocks are filled out 

        #trialType
        ones = np.ones(self.gr1ItemNum)
        twos = np.ones(self.gr2ItemNum)*2
        threes = np.ones(self.gr3ItemNum)*3
        fours = np.ones(self.gr4ItemNum)*4
        fives = np.ones(self.gr5ItemNum)*5

        TTs = np.hstack((ones,twos,threes,fours,fives))
        blockInfo[:,2] = TTs

        #dummies
        TT6 = 0
        TT7 = 0
        TT8 = 0
        TT9 = 0
        TT10 = 0
        TT11 = 0
        TT12 = 0
        TT13 = 0

        #add each TT
        for eachTT in range(5):
            thisTT = self.selectTT(eachTT+1,TT1,TT2,TT3,TT4,TT5,TT6,TT7,TT8,TT9,TT10,TT11,TT12,TT13)

            if eachTT == 0:
                allTTs = thisTT
            else:
                allTTs = np.vstack((allTTs,thisTT))
        blockInfo[:,3:14] = allTTs

        #shuffle trial order
        np.random.shuffle(blockInfo[:,2:14])

        #whichLeft
        whichLeft = np.hstack((np.zeros(int((trialNum)/2)),np.ones(int((trialNum)/2))))
        np.random.shuffle(whichLeft)
        #print(whichLeft)
        blockInfo[:,14] = whichLeft
    
        return blockInfo

    def makeStarBlock(self,eachBlock,star,matchInfo,TT1,TT2,TT3,TT4,TT5,TT6,TT7,TT8,TT9,TT10,TT11,TT12,TT13):
        print('makeStarBlock')

        #item will be tested in 2AFC, correct, and incorrect (hence the x3) and the X items will also be tested once as 2AFC
        trialNum = int((self.gr1ItemNum + self.gr2ItemNum + self.gr3ItemNum + self.gr5ItemNum)*3 + self.gr4ItemNum)
        blockInfo = np.zeros((trialNum,15))

        #blockNum
        blockInfo[:,0] = eachBlock
        #trialNum
        blockInfo[:,1] = 0

        #trialType
        ones = np.ones(self.gr1ItemNum)
        twos = np.ones(self.gr2ItemNum)*2
        threes = np.ones(self.gr3ItemNum)*3
        fours = np.ones(self.gr4ItemNum)*4
        fives = np.ones(self.gr5ItemNum)*5

        for eachScramTT in range(6,14):
            TTsScram = np.ones(self.gr1ItemNum) * eachScramTT

            if eachScramTT == 6:
                allScramTT = TTsScram
            else:
                allScramTT = np.hstack((allScramTT,TTsScram))

        TTs = np.hstack((ones,twos,threes,fours,fives,allScramTT))
        blockInfo[:,2] = TTs

        #add each TT
        for eachTT in range(13):
            thisTT = self.selectTT(eachTT+1,TT1,TT2,TT3,TT4,TT5,TT6,TT7,TT8,TT9,TT10,TT11,TT12,TT13)

            if eachTT == 0:
                allTTs = thisTT
            else:
                allTTs = np.vstack((allTTs,thisTT))
        blockInfo[:,3:14] = allTTs

        #shuffle trial order
        np.random.shuffle(blockInfo[:,2:14])

        #whichLeft
        whichLeft = np.hstack((np.zeros(int((trialNum)/2)),np.ones(int((trialNum)/2))))
        np.random.shuffle(whichLeft)
        #print(whichLeft)
        blockInfo[:,14] = whichLeft

        return blockInfo

    def selectTT(self,thisTT,TT1,TT2,TT3,TT4,TT5,TT6,TT7,TT8,TT9,TT10,TT11,TT12,TT13):
        print("selectTT")

        if thisTT == 1:
            return TT1
        elif thisTT == 2:
            return TT2
        elif thisTT == 3:
            return TT3
        elif thisTT == 4:
            return TT4            
        elif thisTT == 5:
            return TT5
        elif thisTT == 6:
            return TT6
        elif thisTT == 7:
            return TT7
        elif thisTT == 8:
            return TT8
        elif thisTT == 9:
            return TT9
        elif thisTT == 10:
            return TT10
        elif thisTT == 11:
            return TT11
        elif thisTT == 12:
            return TT12
        elif thisTT == 13:
            return TT13

    def makeImagineInfo(self,matchinfo,run):
        print("makeImagineInfo")


        essentialNum1 = self.gr1ItemNum+self.gr2ItemNum+self.gr3ItemNum
        essentialNum2 = self.gr5ItemNum
        matchInfoMin = matchInfo[0:essentialNum1,:]
        matchInfoMin = np.vstack((matchInfoMin,matchInfo[-essentialNum2::,:]))
        matchInfoSh = copy.deepcopy(matchInfoMin) 
        np.random.shuffle(matchInfoSh)

        if run == 9:
            matchInfoPl = matchInfoMin[0:8,:]
            matchInfoPl = np.vstack((matchInfoPl,matchInfoMin[16:24,:]))
            matchInfoPl[:,0] = np.arange(81,97)
            matchInfoPl[:,1] = np.arange(56,72)
            np.random.shuffle(matchInfoPl)
            matchInfoSh = np.vstack((matchInfoSh,matchInfoPl))
    
        row,col = np.shape(matchInfoSh)

        imagineInfo = np.zeros((row,col+1))

        imagineInfo[:,0] = np.arange(row)
        imagineInfo[:,1:(col+1)] = matchInfoSh
       
        np.savetxt(path+"info/imagineInfo/"+str(run)+"/imagineInfo_"+ PID +".csv", imagineInfo, fmt='%i', delimiter=",", header = 'trialNum,matchInfoNum, partWordID, group, countWithin, prefix, imCat,imID')

        return imagineInfo

    def makeJitterInfo(self,imagineInfo,run):
        print("makeJitterInfo")

        row,col = np.shape(imagineInfo)

        jitterInfo = np.zeros((row,8))

        jitterInfo[:,0] = np.arange(row)

        TR = 1.5
        volume = 4
        fixedLength = TR * volume

        jitterInfo[:,1] = TR
        jitterInfo[:,2] = volume
        jitterInfo[:,3] = fixedLength

        jitter0 = np.zeros(int(row/2))
        jitter1 = np.ones(int(row/2)) * TR
        jitter = np.hstack((jitter0,jitter1))
        np.random.shuffle(jitter)

        jitterInfo[:,4] = jitter

        waitTime = 0.2
        waitTime = np.ones(row) * waitTime
        fixedLength = np.ones(int(row))*fixedLength
        startLooking = fixedLength+jitter - waitTime


        jitterInfo[:,5] = startLooking

        #approximate trialLength since trigger may come a bit before or after
        trialLength = fixedLength+jitter
        jitterInfo[:,6] = trialLength

        #approximate end time for that trial for the given run
        expTime = np.zeros(row)

        for eachTrial in range(row):

            thisTrialLength = jitterInfo[eachTrial,6]

            if eachTrial == 0:
                thisExpTime = thisTrialLength
            else:
                thisExpTime = thisExpTime+thisTrialLength

            expTime[eachTrial] = thisExpTime

        jitterInfo[:,7] = expTime

        np.savetxt(path+"info/jitterInfo/"+str(run)+"/jitterInfo_"+ PID +".csv", jitterInfo, fmt='%f', delimiter=",", header = 'trialNum, TR, volume, fixedLength, jitter, startLooking, trialLength, expTime')

        return

    def makeLocInfo(self):
        print("makeLocInfo")

        #information specific to localizer info
        trialNum = 10
        catNum = 3
        catRepeat = 3
        blockNum =  catNum * catRepeat
        runNum = 3
        totalTrialNum = trialNum * blockNum * runNum

        time_betweenBlock = 15
        time_image = 1.5

        locInfo = np.zeros((totalTrialNum,8))

        #trial
        locInfo[:,0] = np.arange(totalTrialNum)

        #block
        block_run = np.arange(blockNum)
        block_run = np.repeat(block_run,trialNum)
        block_all = np.tile(block_run,runNum)
        locInfo[:,2] = block_all

        #run
        run_all = np.arange(runNum)
        run_all = np.repeat(run_all,blockNum*trialNum)
        locInfo[:,1] = run_all


        #run

        for eachRun in range(3):

            #imCat
            imCat_types = np.arange(catNum) # for the three categories
            imCat_types = np.repeat(imCat_types,catRepeat)# for the 3 blocks each
            np.random.shuffle(imCat_types) #to mix the order of blocks randomly
            imCat_all = np.repeat(imCat_types,trialNum) # for the 10 trials each

            begin = eachRun * (trialNum*blockNum)
            end = begin + (trialNum*blockNum)
            #print(eachRun,begin,end)

            #print(begin,end,np.shape(imCat_all))
            locInfo[begin:end,3] = imCat_all


            #repeat & onset
            repeat = np.zeros(trialNum*blockNum)
            onset = np.zeros(trialNum*blockNum)

            for eachBlock in range(blockNum):
                #repeat
                nonScramble = np.array([0,0])
                scramble = np.array([0,0,0,0,0,0,1,1])
                np.random.shuffle(scramble)
                thisRepeat = np.hstack((nonScramble,scramble))

                #onset
                thisOnset = np.arange(trialNum)
                thisOnset = thisOnset * time_image
                thisOnset = thisOnset + (time_betweenBlock * (eachBlock * 2 +1))

                begin_bl = eachBlock * trialNum
                end_bl = begin_bl + trialNum

                #print(begin,end,thisRepeat)
                #print(np.shape(thisRepeat))
                repeat[begin_bl:end_bl] = thisRepeat
                onset[begin_bl:end_bl] = thisOnset

            locInfo[begin:end,4] = repeat
            locInfo[begin:end,6] = onset

            offset = onset + 1.5
            locInfo[begin:end,7] = offset

            #imID
            faceID = 0
            sceneID = 0
            objID = 0

            for eachTrial in range(totalTrialNum):
                imCat = locInfo[eachTrial,3]
                repeat = locInfo[eachTrial,4]

                if repeat == 0: 

                    if imCat == 0:
                        faceID += 1
                        imID = faceID

                    elif imCat == 1:
                        sceneID += 1
                        imID = sceneID

                    elif imCat == 2:
                        objID += 1
                        imID = objID

                locInfo[eachTrial,5] = imID

        np.savetxt(path+"info/locInfo/locInfo_"+ PID +".csv", locInfo, fmt='%i,%i,%i,%i,%i,%i,%f,%f', delimiter=",", header = 'trial,run,block,imCat,repeat,imID,targetOnset,targetOffset')

        return 

    def makeItemtestInfo(self,matchInfo):
        print("makeItemtestInfo")


        trialNum = self.gr1ItemNum + self.gr2ItemNum + self.gr3ItemNum + self.gr5ItemNum

        testInfo = np.zeros((trialNum,15))

        for eachGroup in range(4):
            if eachGroup == 0:
                option1 = np.arange(self.gr1ItemNum)
                option2 = np.arange(self.gr1ItemNum)
            elif eachGroup == 1:
                option1 = np.arange(self.gr2ItemNum) + 8
                option2 = np.arange(self.gr2ItemNum) + 8
            elif eachGroup == 2:
                option1 = np.arange(self.gr3ItemNum) + 16
                option2 = np.arange(self.gr3ItemNum) + 16          
            elif eachGroup == 3:
                option1 = np.arange(self.gr5ItemNum) + 48
                option2 = np.arange(self.gr5ItemNum) + 48   

            np.random.shuffle(option1)
            overLap = True

            while overLap == True:
                np.random.shuffle(option2)
                overLap = False

                for eachItem in range(self.gr1ItemNum):
                    if option1[eachItem] == option2[eachItem]:
                        #print(eachItem,option2)
                        overLap = True

            choices = np.vstack((option1,option1))
            choices = np.vstack((choices,option2))
            choices = np.transpose(choices)

            #print("eachGroup",eachGroup)
            #print("choices",choices)
            choiceInfo = self.getDetails(self.gr1ItemNum,choices,matchInfo)
            np.random.shuffle(choiceInfo)
            testInfo[(eachGroup*8):(eachGroup*8+8),3:14] = choiceInfo

            #trialType
            testInfo[(eachGroup*8):(eachGroup*8+8),2] = 14+eachGroup

        np.random.shuffle(testInfo)

        #blockNum
        testInfo[:,0] = 0

        #testNum
        testInfo[:,1] = np.arange(trialNum)

        #whichLeft
        whichLeft = np.hstack((np.zeros(int(trialNum/2)),np.ones(int(trialNum/2))))
        np.random.shuffle(whichLeft)
        testInfo[:,14] = whichLeft
       
        np.savetxt(path+"info/itemtestInfo/itemtestInfo_"+ PID +".csv", testInfo, fmt='%i', delimiter=",", header = 'blockNum,trialNum,trialType,matchInfoNumAud,prefix,pwID,matchInfoNum1,matchInfoNum2,group type1,group type 2,imCat1,imCat2,imID1,imID2,whichLeft')

        return testInfo






exp = gen_info()
matchInfo,matchInfoExtra = exp.makeMatchInfo(condRow,condDef)
exp.makeExpoInfo(matchInfo)
exp.makeRunInfo(matchInfoExtra)
imagineInfo4 = exp.makeImagineInfo(matchInfo,4)
imagineInfo5 = exp.makeImagineInfo(matchInfo,5)
imagineInfo8 = exp.makeImagineInfo(matchInfo,8)
imagineInfo9 = exp.makeImagineInfo(matchInfo,9)
exp.makeJitterInfo(imagineInfo4,4)
exp.makeJitterInfo(imagineInfo5,5)
exp.makeJitterInfo(imagineInfo8,8)
exp.makeJitterInfo(imagineInfo9,9)
exp.makeLocInfo()
exp.makeItemtestInfo(matchInfo)





