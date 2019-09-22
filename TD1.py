from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import random
from statistics import mode

from math import sqrt

def loadfile(filename, class_index):
  '''Loads data from .data file and separes it in features and labels 

  input:
  filename: path of the file to be analysed
  class_index: index of where are the labels of the data

  output: 
  x: features of the data - desconsidering ID for example
  y: labels '''

  f= open(filename,'r') 
  lines = f.readlines()
  y=[]
  x=[]
  identity=[]
  subset=[]
  for line in lines[0:]:
     subset=line.split(',')

     '''In case of the breast-cancer-wisconsin data set there is an ID atributed to each patient- 
	as the ID its not a tumor characteristic, it's also removed '''

     if(filename=="breast-cancer-wisconsin.data"):
       identity.append(subset.pop(0))


     y.append(subset.pop(class_index).replace('\n','')) 
     x.append(subset)
  print('Number of instances '+ str(np.size(y)))
  return x,y


def split_data(x,y,train_fraction):
  '''Divides data set randomly in the subsets test and train

  input:
  x: data features
  y: data labels
  train_fraction: value from 0 to 1 that represents the porcentage of data
  used to train the kNN

  output: 
  x_test: features of the testing data 
  y_test: labels of the testing data
  x_train: features of the training data
  y_data: labels of the trainning data '''

  x_train=[]
  y_train=[]
  x_test=[]
  y_test=[]
 
  train_size=int(np.size(y)*train_fraction)

  y_test=y
  x_test=x

  for i in range (train_size):
    index=randint(0, np.size(y)-1)

    y_train.append(y_test[index])
    x_train.append(x_test[index])
    y_test.pop(index)
    x_test.pop(index)
 

  return x_test, y_test, x_train, y_train

def characters_treatement (character, x1, x2):
  '''Data pre treatement to deal with character apereances
    
  input: 
  character: char that is in the data set 
  x1: feature
  x2: feature

  output: 
  true/false statement'''
  if(x1==character or x2==character):
    return True
  else: 
    return False


def distance_euclidian(x1, x2):

  '''Calculates the euclidian distances between two features list

  input: 
  x1: features list
  x2: features list

  output: 
  d: euclidian distance'''
  d=0

  for i in range(len(x2)):
    '''There is a data set where ? characters where found in order to treat them,
       we consider that the distances is null, there is to say they will have no 
    impact in the distance total'''
    if(characters_treatement('?', x1[i], x2[i])):
      d+=0
    else:
      d+=(float(x1[i])-float(x2[i]))**2
  d=sqrt(d)
  return d

def distance_manhattan(x1, x2):
  '''Calculates the manhattan distances between two features list

  input: 
  x1: features list
  x2: features list

  output: 
  d: manhattan distance'''

  d=0
  for i in range(len(x2)):
    if(characters_treatement('?', x1[i], x2[i])):
      d+=0
    else:
      d+=(float(x1[i])-float(x2[i]))
  return d



def distance_chebychev(x1, x2):

  '''Calculates the chebychev distances between two features list

  input: 
  x1: features list
  x2: features list

  output: 
  d: chebychev distance'''

  d=[]
  for i in range(len(x2)):
    if(characters_treatement('?', x1[i], x2[i])):
      d.append(0)
    else:
      d.append(float(x1[i])-float(x2[i]))
  return max(d)



def distance_hamming(x1, x2):

  '''Calculates the hamming distances between two features list

  input: 
  x1: features list
  x2: features list

  output: 
  d: hamming distance'''

  d=0
  for i in range(len(x2)):
    if(characters_treatement('?', x1[i], x2[i])):
      d+=0
    else:
      if float(x1[i])-float(x2[i])!=0:
        d+=1
      else:
        d+=0
  return d


  
def confusion_calcul(y_test, y_pred, class1, class2):

  '''Calculates the confusion matrix of a prediction

  input: 
  y_test: test data set labels
  y_pred: predicted test data set labels
  class1: possible class/label 
  class2: possible class/label

  output: 
  confusion_matrix: confusion matrix of a prediction'''

  confusion_matrix=[[0,0], [0,0]]
  for i in range(len(y_test)):
    if float(y_pred[i])==float(class1):
      if y_pred[i]==y_test[i]:
        confusion_matrix[0][0]=confusion_matrix[0][0]+1
      else:   
        confusion_matrix[1][0]=confusion_matrix[1][0]+1
    if float(y_pred[i])==float(class1):
      if y_pred[i]==y_test[i]:
        confusion_matrix[1][1]=confusion_matrix[1][1]+1
      else:
        confusion_matrix[0][1]=confusion_matrix[0][1]+1
  return confusion_matrix


     
def accuracy_calcul(y_test, y_pred): 

  '''Calculates the accuracy of a prediction
  
  input: 
  y_test: test data set labels
  y_pred: predicted test data set labels

  output: 
  accuracy: accuracy of the prediction (between 0-1)'''

  accuracy=0
  for i in range(len(y_test)):
    if y_pred[i]==y_test[i]:
      accuracy=accuracy+1
  accuracy=float(accuracy)/np.size(y_test)
  return accuracy
      


def kNN(x_train, y_train, x_test, k, distance):

  '''Predicts the labels of the test dataset according with the train data set 

  
  input: 
  x_train:train data set features
  y_train:train data set labels
  x_test: test data set features
  k: number of neighbors to be considered
  distance: type of distance to be considered

  output: 
  y_pred: predicted labels of the test data set'''


  y_pred=[]
  vote=[]
  for test in x_test:
    distances=[]
    for train in x_train:
      distances.append(distance(train, test))
    index=np.argsort(distances)
    for i in range(k):
      vote=y_train[index[i]]
    y_pred.append(mode(vote))
  return y_pred 
      
      
def choose_property(property1):
   '''For the breast-cancer-wisconsis allows to choose the features in a natural language'''
   switcher={
            "Clump Thickness": 0,    
            "Uniformity of Cell Size":1,
            "Uniformity of Cell Shape":2,
            "Marginal Adhesion":3,
            "Single Epithelial Cell Size": 4,
            "Bare Nuclei": 5,
            "Bland Chromatin":6,
            "Normal Nucleoli":7,
            "Mitoses":8,
        }
   return switcher.get(property1, "Not Found") 

def plot_data(x, y, property1, property2):
    '''Plots the comparison between 2 features from the breast- cancer-wisconsin data set 

    input: 
    x: features of a data set
    y: label of a data set 
    property1: property to be viewed
    property1: property to be viewed

    output: 
    graph: graph with the comparison between the two features of the tumor showing if it was benign or not'''


    p1=choose_property(property1)
    x_b1=[]
    x_b2=[]
    x_m1=[]
    x_m2=[]
    p2=choose_property (property2)
    for i in range(len(x)-1):
      for l in range(len(x[i])-1):
        if(x[i][l]=='?'):
          x[i][l]=0
      if(float(y[i])==2):
         x_b1.append(x[i][p1])
         x_b2.append(x[i][p2])
      else:
         x_m1.append(x[i][p1])
         x_m2.append(x[i][p2])
    ax=plt.scatter(x_b1, x_b2, s=100, c="blue", label='benign')
    ax=plt.scatter(x_m1, x_m2, s=100, c="red", label='malign')
    plt.title(property1+'vs'+property2)
    plt.xlabel(property1)
    plt.ylabel(property2)
    plt.grid(color='gray', linestyle='-', linewidth=1)
    plt.legend()
    plt.show()

def plot_data3d(x, y):

    '''Plots the comparison between the 3 features from the haberman data set 

    input: 
    x: features of a data set
    y: label of a data set 
    

    output: 
    graph: graph 3d with axis being the 3 features'''

    x1=[]
    x2=[]
    x3=[]
    x4=[]
    x5=[]
    x6=[]
   
    for i in range (len(y)):
      if( y[i]=='1'):
        x1.append(int(x[i][0]))
        x2.append(int(x[i][1]))
        x3.append(int(x[i][2]))
      else:
        x4.append(int(x[i][0]))
        x5.append(int(x[i][1]))
        x6.append(int(x[i][2]))
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x1, x2, x3, c="blue", s=100,label='Survived 5 years or longer')
    ax.scatter(x4, x5, x6, c="red", s=100,label='Did not survived 5 years or longer')
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Surgery year')
    ax.set_zlabel('Positive axillary nodes')
    plt.legend()
    plt.show()


def distances_comparison(x_train, y_train, x_test, k):
  '''Calculs the accuracy of each distances and returns the best prediction

    input: 
    x_train: features of the train data set
    y_train: label of the train data set 
    x_test: features of the train data set
    k: number of neighbors

    output: 
    y_pred:best predicted label for the test data set '''

  distances=[distance_euclidian, distance_manhattan, distance_chebychev, distance_hamming]
  y_pred=[]
  acc=[]
  y=[]
  for distance in distances:
    y_pred=((kNN(x_train, y_train, x_test, k, distance_euclidian)))
    y.append(y_pred)
    acc.append(accuracy_calcul(y_test, y_pred))
  
  index=np.where(max(acc))
  print(index)
  plot_accuracy(acc)
  return y[index[0]] 
 
def plot_accuracy(acc):
  '''Plots the accuracy of each distances

    input: 
    acc: list of accuracies

    output: 
    graph: bar graph with each accuracy (%) '''
  
  plt.bar(1, acc[0]*100, color='blue', label='Euclidian')
  plt.bar(2, acc[1]*100, color='#6A5ACD', label='Manhattan')
  plt.bar(3, acc[2]*100, color='#6495ED',label='Chebychev')
  plt.bar(4, acc[3]*100, color='#00BFFF', label='Hamming')
  plt.title('Accuracy vs Distance types')
  plt.ylabel('Accuracy(%)')
  plt.legend()
  plt.show()
   

'''Main for the data file  breast-cancer-wisconsin.data'''
x,y=loadfile("breast-cancer-wisconsin.data", 9)
x_test, y_test, x_train, y_train=split_data(x,y,.5)
y_pred=distances_comparison(x_train, y_train, x_test, 5)
plot_data(x_train, y_train, "Bare Nuclei","Mitoses")
print(confusion_calcul(y_test, y_pred, 2, 4))


'''Main for the data file  hammerman.data'''

x,y=loadfile("haberman.data", 3)
x_test, y_test, x_train, y_train=split_data(x,y,.5)
y_pred=distances_comparison(x_train, y_train, x_test, 5)
print(confusion_calcul(y_test, y_pred, 1, 2))
plot_data3d(x_test, y_pred)

