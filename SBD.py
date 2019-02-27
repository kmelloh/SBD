"""

--- Sentence Boundary Detection ---

This is a program that detects whether a period is an end of sentence or not depending
on a set of features. The features are retrieved for each period,
and are then fed into a classifier (decision tree).
The decision tree generates the result,and the result is then compared to the test data set.

The program generates a file called SBD.test.out, which displays the word that has the period next to it,
the predicted tag, and the actual tag

"""

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import sys
import collections



def isLessthan3(i):
    return i<3

def isCapital(word):
    return word.istitle()

def isLower(word):
    return not word.istitle()

def countLeftPeriod(word):
   return wordCounter[word]

def countRightLower(word):
    return wordCounter[word]
def countLeftNoPeriod(word):
    return wordCounter[word[:-1]]

    

training_file=open(sys.argv[1],"r")
test_file=open(sys.argv[2],"r")



training_dataset=training_file.read().split("\n")
test_dataset=test_file.read().split("\n")
training_features=[]
test_features=[]
training_target=[]
test_target=[]
wordsList=[]




for item in training_dataset:
    try:
        wordsList.append(item.split()[1])
    except IndexError:
        pass




wordsList=np.array(wordsList)
wordCounter=collections.Counter(wordsList)



for item in training_dataset:        # remove special characters from the training data set, mainly commas
    try:
        if(item.split()[1]=="," or item.split()[1]=="\"" or item.split()[1]=="\'" or item.split()[1]=="!" or item.split()[1]=="?"):
           del training_dataset[training_dataset.index(item)]
           
    except ValueError:
        pass
    except IndexError:
        pass
    





for item in training_dataset:     # ignore items with the TOK tag, and add the other items to training_features
     try:
        if(item.split()[2]!="TOK"):
            leftWord=item.split()[1]
            rightWord=training_dataset[training_dataset.index(item)+1].split()[1]
            tag=item.split()[2]
            training_features.append(([leftWord.replace(".",""),rightWord,isLessthan3(len(leftWord)),isCapital(leftWord),isCapital(rightWord),int((countLeftPeriod(leftWord))),int((countRightLower(rightWord.lower()))),int(countLeftNoPeriod(leftWord))]))
            training_target.append(tag)              #create a list that has the training dataset target features in it(NEOS or EOS)
#
     except IndexError:
        pass
    


training_target=[0 if item=="NEOS" else 1 for item in training_target]    #transform the target features list to 0s and 1s (0 --> NEOS, 1-->EOS)


training_features=np.array(training_features)




    
    




encoder=preprocessing.LabelEncoder()                   #transform the values in the training features list to numbers(used to train the classifier)
encoded_training_features=np.array(training_features)

encoded_training_features=encoder.fit_transform(encoded_training_features.flatten())
encoded_training_features=np.reshape(encoded_training_features,(-1,8))


for item in test_dataset:       # remove special characters from the training data set, mainly commas
    try:
        if(item.split()[1]=="," or item.split()[1]=="\"" or item.split()[1]=="\'" or item.split()[1]=="!" or item.split()[1]=="?"):
           del test_dataset[test_dataset.index(item)]
    except ValueError:
        pass
    except IndexError:
        pass
   
        
for item in test_dataset:        # ignore items with the TOK tag, and add the other items to test_features
     try:
        if(item.split()[2]!="TOK"):
            leftWord=item.split()[1]
            rightWord=test_dataset[test_dataset.index(item)+1].split()[1]
            tag=item.split()[2]
            test_features.append([leftWord.replace(".",""),rightWord,isLessthan3(len(leftWord)),isCapital(leftWord),isCapital(rightWord),int((countLeftPeriod(leftWord))),int((countRightLower(rightWord.lower()))),int(countLeftNoPeriod(leftWord))])
            test_target.append(tag)            #create a list that has the test dataset target features in it(NEOS or EOS)
           
#
     except IndexError:
        pass
    

test_target=[0 if item=="NEOS" else 1 for item in test_target]      #transform the target features list to 0s and 1s (0 --> NEOS, 1-->EOS)
encoded_test_features=np.array(test_features)

encoded_test_features=encoder.fit_transform(encoded_training_features.flatten())



encoded_test_features=np.reshape(encoded_test_features,(-1,8))  #reshape the encoded test features list into rows of 5 columns each. This is useful for our prediction
training_target=np.array(training_target)


tree=DecisionTreeClassifier()                       #create the decision tree
tree.fit(encoded_training_features,training_target)     #feed the classifier the training data

prediction=tree.predict(encoded_test_features)      #predict what the target values will look like based on what we fed the classifier

prediction=prediction[:len(test_target)]            #shortening the length of the predictions to match the test_target, in order to be able to calculae the score
score=accuracy_score(test_target,prediction)
print("The accuracy of the system is: ",round(score*100,2),"%")


prediction=["NEOS" if str(item)=="0" else "EOS" for item in prediction]  #reverting the encoding done
test_target=["NEOS" if str(item)=="0" else "EOS" for item in test_target]




outputFile=open("SBD.test.out","w")
outputFile.write("Number \t Word \t Predicted Tag \t Actual Tag\n")
outputFile.write("---------------------------------------------\n")


for item,target in zip(test_dataset,prediction):


   if(item[-3:]!="TOK"):
        outputFile.write("{:11}{:11}{:11}{:11}".format(item.split()[0],item.split()[1],target,item.split()[2])) 
        
        item.replace(item[-3:],target)
        outputFile.write("\n")
        
        
    
outputFile.close()





training_file.close()
test_file.close()