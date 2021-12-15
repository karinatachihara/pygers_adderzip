# -*- coding: utf-8 -*-
"""
Created on 09/16/19

This takes separate prefix and suffixes 

"""
#import libraries
import numpy as np
from pydub import AudioSegment

path = "/Users/Work/Documents/ResearchProject/adderzip/exp_fMRI/stimuli/"

for eachWord in range(83):
	pw = path+"partWords_wav/pw_"+str(eachWord)+".wav"
	abber = path+"prefix_wav/abber.wav"
	belling = path+"prefix_wav/belling.wav"

	pw = AudioSegment.from_wav(pw)
	abber = AudioSegment.from_wav(abber)
	belling = AudioSegment.from_wav(belling)

	abberWord = abber+pw
	bellingWord = belling+pw

	abberWord.export(path+"partWords_wav/abber/abberWord"+str(eachWord)+".wav",format = 'wav')
	bellingWord.export(path+"partWords_wav/belling/bellingWord"+str(eachWord)+".wav",format = 'wav')
