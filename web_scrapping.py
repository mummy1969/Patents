

import numpy as np
import tensorflow as tf
import re
import time

#Loading the dataset's conversation and lines text
lines=open('movie_lines.txt', encoding='utf-8',errors= 'ignore').read().split('\n')
conversations=open('movie_conversations.txt', encoding='utf-8',errors= 'ignore').read().split('\n')

id2line = {}
for line in lines:
    _line=line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
conversation_ids=[]
for conversation in conversations[:-1]:
    _conversation=conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversation_ids.append(_conversation.split(","))
#creating questions and answers
    
questions = []
answers = []
for conversation in conversation_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
#creating a function for cleaning the text using it as questions and answers text
def clean_text(text):
    text=text.lower()
    text=re.sub(r"i'm","i am", text)
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"she's","she is",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"where's","where is",text)
    text=re.sub(r"that's","that is",text)
    text=re.sub(r"won't","will not",text)
    text=re.sub(r"\'d"," would",text)
    text=re.sub(r"\'ll"," will",text)
    text=re.sub(r"\'ve"," have",text)
    text=re.sub(r"\'re"," are",text)
    text=re.sub(r"can't","cannot",text)
    text=re.sub(r"[-()\+=?/,.:;<>{}@#~]","",text)
    return text

#creating cleaned questions list                
clean_questions=[]
for question in questions:
    clean_questions.append(clean_text(question))
#creating cleaned answers list  
clean_answers=[]
for answer in answers:
    clean_answers.append(clean_text(answer))            
#creating a dictionary that maps word to number of occurences in both cleaned questions and answers
word2count={}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
#creating a dictionary for questions and answers that will map it to a unique integer
threshold=20
questionswords2int={}
word_number=0
for word, count in word2count.items():
    if count>=threshold:
        questionswords2int[word]=word_number
        word_number+=1
answerswords2int={}
word_number=0
for word, count in word2count.items():
    if count>=threshold:
        answerswords2int[word]=word_number

tokens=['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token]=len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token]=len(answerswords2int) + 1
#creating an inverse dictionary for answerswords2int
answersints2word={w_i: w for w, w_i in answerswords2int.items()}