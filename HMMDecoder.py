#!/usr/bin/python2

# Author: Deepak Pandita
# Date created: 06 Sep 2017

import math
import json
import numpy as np

train_file = '/data/train'
test_file = '/data/test'
#transition probabilities file
A_file = 'A'
#emission probabilities file
B_file = 'B'


#Observation given tag
OT = {}
#Tag given tag
TT = {}
#All tags
T = {}
#All words
W = {}

#Read train file
print 'Reading file: '+train_file

with open(train_file) as tf:
	lineNum = 0
	for line in tf:
		
		tokens = line.strip().split(' ')[1:]
		
		words = [y for x,y in enumerate(tokens) if x%2 == 0]
		tags = [y for x,y in enumerate(tokens) if x%2 != 0]
		#print words
		#print tags
		
		#Calculate frequencies
		for word in words:
			if W.has_key(word):
				freq = W[word]
				W[word] = freq+1
			else:
				W[word] = 1
		index = 0
		prevTag = ''
		for tag in tags:
			if T.has_key(tag):
				freq = T[tag]
				T[tag] = freq+1
			else:
				T[tag] = 1
			if OT.has_key(words[index]+'_'+tag):
				freq = OT[words[index]+'_'+tag]
				OT[words[index]+'_'+tag] = freq+1
			else:
				OT[words[index]+'_'+tag] = 1
			if index!=0:
				if TT.has_key(tag+'_'+prevTag):
					freq = TT[tag+'_'+prevTag]
					TT[tag+'_'+prevTag] = freq+1
				else:
					TT[tag+'_'+prevTag] = 1
			prevTag = tag
			index = index + 1
		lineNum = lineNum + 1
		if lineNum%5000==0:
			print 'Read '+str(lineNum)+' lines'
	print 'Total lines read: '+str(lineNum)
#print T
#print TT
#print OT

A = {}
B = {}

print 'Calculating probabilities...'
#Calculate probabilities
for key, value in TT.items():
		tag = key.split('_')[1]
		prob = float(value)/T[tag]
		A['A_'+key] = math.log(prob)	#taking log probability

for key, value in OT.items():
		tag = key.split('_')[1]
		prob = float(value)/T[tag]
		B['B_'+key] = math.log(prob)	#taking log probability

#print A
#print B

print 'Writing files...'
#Store the transition and emission probabilities
with open(A_file, 'w') as af:
	json.dump(A,af)

with open(B_file, 'w') as bf:
	json.dump(B,bf)

print 'Len A: ' + str(len(A))
print 'Len B: ' + str(len(B))
print 'Len T: ' + str(len(T))
print 'Len W: ' + str(len(W))


#Viterbi Algorithm
def viterbi(A,B,test_sequence,all_tags):
	best_tag_sequence = []
	probability_matrix = np.ones((len(all_tags),len(test_sequence)))*float('-inf')
	backpointer = np.zeros((len(all_tags),len(test_sequence)))
	for index,obs in enumerate(test_sequence):
		for tag_index,tag in enumerate(all_tags):
			if index==0:
				probability_matrix[tag_index,index] = B.get('B_'+obs+'_'+tag,float('-inf'))
				backpointer[tag_index,index] = -1
			else:
				probabilities = []
				b = B.get('B_'+obs+'_'+tag,-1*1e5)
				for prev_index,prev_tag in enumerate(all_tags):
					a = A.get('A_'+tag+'_'+prev_tag,-1*1e5)
					prob = probability_matrix[prev_index,index-1]+a+b;	#summing because of log probabilities
					probabilities.append(prob)
				probability_matrix[tag_index,index] = max(probabilities)	#get max probability
				backpointer[tag_index,index] = np.argmax(probabilities)		#get argmax
	#Best score = max probability in the last column of the probability matrix
	#best_score = max(probability_matrix[:,-1])
	
	#The start of backtrace = index of max probability in the last column of the probability matrix
	backtrace = np.argmax(probability_matrix[:,-1])
	
	for prev_best_tag in xrange(np.size(backpointer,axis=1),0,-1):
		best_tag_sequence.append(all_tags[int(backtrace)])
		backtrace = backpointer[int(backtrace),prev_best_tag-1]

	best_tag_sequence.reverse()
	return best_tag_sequence

#Read test file
print 'Reading file: '+test_file

correct_tags = 0
total_tags = 0
with open(test_file) as tf:
	lineNum = 0
	for line in tf:
		tokens = line.strip().split(' ')[1:]
		words = [y for x,y in enumerate(tokens) if x%2 == 0]
		tags = [y for x,y in enumerate(tokens) if x%2 != 0]
		#Call viterbi algorithm
		best_tag_sequence = viterbi(A,B,words,T.keys())
		#print words
		#print tags
		#print best_tag_sequence
		for x in range(len(tags)):
			if tags[x]==best_tag_sequence[x]:
				correct_tags = correct_tags + 1
		total_tags = total_tags + len(tags)
		
		lineNum = lineNum + 1
		if lineNum%200==0:
			print 'Read '+str(lineNum)+' lines'
	print 'Total lines read: '+str(lineNum)

print 'Correct tags: '+str(correct_tags)+' Total tags: '+str(total_tags)
print 'Accuracy: ' + str(float(correct_tags)/total_tags)
print 'Done'