# For generating User 1 data 

import json
import random
list1=[]
with open("MultiWOZ.json", "r") as read_file:
    data = json.load(read_file)
for data1 in data:
    list1.append("\n\n")
    i=0
    while i < len(data[data1]["log"]):
        char1=data[data1]["log"][i]['text']
        list1.append( char1 + " <ECON> ")
        i=i+2
f= open("/data/new_data_user1.txt","w+")
f.write("".join(list1))
f.close()


# For generating User 2 data 

import json
import random
list1=[]
with open("MultiWOZ.json", "r") as read_file:
    data = json.load(read_file)
for data1 in data:
    list1.append("\n\n")
    i=1
    while i < len(data[data1]["log"]):
        char1=data[data1]["log"][i]['text']
        list1.append( char1 + " <ECON> ")
        i=i+2
f= open("/data/new_data_user2.txt","w+")
f.write("".join(list1))
f.close()


# For generating User 1 data by splitting the whole message where message may contain one scentence and some remiang


import json
import random
list1=[]
with open("MultiWOZ.json", "r") as read_file:
    data = json.load(read_file)
for data1 in data:
    list1.append("\n\n")
    i=0
    while i < len(data[data1]["log"]):
        char1=data[data1]["log"][i]['text']
        words1=char1.split()
        random_no_of_words=random.randint(int(len(words1)/2),len(words1)-1)
        appended_words = " ".join(words1[:random_no_of_words])
        list1.append( appended_words + " <ECON> ")
        i=i+2
f= open("/data/new_data_negative_user1.txt","w+")
f.write("".join(list1))
f.close()

# For generating User 2 data by splitting the whole message where message may contain one scentence and some remianing part

import json
import random
list1=[]
with open("/Users/sriramvithala/Downloads/NLP project data/MultiWOZ_1/MultiWOZ.json", "r") as read_file:
    data = json.load(read_file)
for data1 in data:
    list1.append("\n\n")
    i=1
    while i < len(data[data1]["log"]):
        char1=data[data1]["log"][i]['text']
        words1=char1.split()
        random_no_of_words=random.randint(int(len(words1)/2),len(words1)-1)
        appended_words = " ".join(words1[:random_no_of_words])
        list1.append( appended_words + " <ECON> ")
        i=i+2
f= open("/data/new_data_negative_user2.txt","w+")
f.write("".join(list1))
f.close()


# For generating User 1 data by splitting the message where not even a single message has been completed or sent


import json
import random
list1=[]
with open("MultiWOZ.json", "r") as read_file:
    data = json.load(read_file)
for data1 in data:
    list1.append("\n\n")
    i=0
    while i < len(data[data1]["log"]):
        char1=data[data1]["log"][i]['text']
        words=char1.split()
        words1=[]
        while words:
            word=words.pop(0)
            words1.append(word)
            if word=="." or word[-1]=='.' or word=="?" or word[-1]=='?' or word=="!" or word[-1]=='!':
                break
        random_no_of_words=random.randint(int(len(words1)/2),len(words1)-1)
        appended_words = " ".join(words1[:random_no_of_words])
        list1.append( appended_words + " <ECON> ")
        i=i+2
f= open("/data/new_data_negative_user1_type2_first_scentence_incomplete.txt","w+")
f.write("".join(list1))
f.close()


# For generating User 2 data by splitting the message where not even a single message has been completed or sent


import json
import random
list1=[]
with open("MultiWOZ.json", "r") as read_file:
    data = json.load(read_file)
for data1 in data:
    list1.append("\n\n")
    i=1
    while i < len(data[data1]["log"]):
        char1=data[data1]["log"][i]['text']
        words=char1.split()
        words1=[]
        while words:
            word=words.pop(0)
            words1.append(word)
            if word=="." or word[-1]=='.' or word=="?" or word[-1]=='?' or word=="!" or word[-1]=='!':
                break
        random_no_of_words=random.randint(int(len(words1)/2),len(words1)-1)
        appended_words = " ".join(words1[:random_no_of_words])
        list1.append( appended_words + " <ECON> ")
        i=i+2
f= open("/data/new_data_negative_user2_type2_first_scentence_incomplete.txt","w+")
f.write("".join(list1))
f.close()



