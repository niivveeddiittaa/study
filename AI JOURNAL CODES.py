#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##practical 1 ss1


# In[1]:


import random
OPEN=['S']
map_list={'S':['A','B','C'],
          'A':['S','D'],
         'B':['S','E'],
         'C':['S','F'],
         'D':['A','G'],
         'E':['B','G','F'],
         'F':['C','E'],
          'G':['D','E']}
def modegen(node):
    return map_list[node]
def goaltest(node):
    return node=='G'
def ss1():
    while len(OPEN)>0:
        random.shuffle(OPEN)
        N=OPEN.pop()
        if goaltest(N):
            return "!!!found!!!"
        else:
            n=modegen(N)
            for i in n:
                if n not in OPEN:
                    OPEN.append(i)
                print("OPEN_LIST",OPEN)
    return"NAI MILA"
print(ss1())


# In[ ]:


##practical 2 ss2


# In[2]:


import random
OPEN=['S']
CLOSED=[]
map_list={'S':['A','B','C'],
          'A':['S','D'],
         'B':['S','E'],
         'C':['S','F'],
         'D':['A','G'],
         'E':['B','G','F'],
         'F':['C','E'],
        'G':['D','E']}
def modegen(node):
    return map_list[node]
def goaltest(node):
    return node=='G'
def ss2():
    while len(OPEN)>0:
        random.shuffle(OPEN)
        N=OPEN.pop()
        CLOSED.append(N)
        if goaltest(N):
            return "found"
        else:
            n=modegen(N)
            for i in n:
                if i not in CLOSED and i not in OPEN:
                    OPEN.append(i)
                    print("OPEN_LIST",OPEN)
                    print("CLOSED_LIST",CLOSED)
    return"NOT FOUND"
print(ss2())


# In[ ]:


##practical 3 ss3


# In[3]:


import random
OPEN=[['S',None]]
CLOSED=[]
map_list={'S':['A','B','C'],
          'A':['S','D'],
          'B':['S','E'],
          'C':['S','F'],
          'D':['A','G'],
          'E':['B','G','F'],
          'F':['C','E'],
          'G':['D','E']
          }
def movegen(node):
    return map_list[node]
def goaltest(node):
    return node=='G'
def returnpath(path):
    if path is not None:
        return str(path[0])+returnpath(path[1])
    else:
        return ""
def ss3():
    while len(OPEN)>0:
        random.shuffle(OPEN)
        print("OPEN_LIST",OPEN)
        M=OPEN.pop()
        N=M[0]
        CLOSED.append(N)
        print("Picked: ",CLOSED)
        if goaltest(N):
            print("GOAL Found")
            print("Path: ",returnpath(M)[::-1])
            return N
        else:
            neigh=movegen(N)
            for node in neigh:
                if node not in CLOSED and node not in OPEN:
                    new_list=[node,M]
                    OPEN.append(new_list)
    print("Not Found")
print(ss3())


# In[6]:


##practical 4 breadth first search


# In[4]:


#BFS#
graph={'S':['A','B','C'],
        'A':['S','D'],
         'B':['S','E'],
         'C':['S','F'],
         'D':['A','G'],
         'E':['B','G','F'],
         'F':['C','E'],
          'G':['D','E']}
visited=[]
queue=[]
def bfs(visited,graph,node):
    visited.append(node)
    queue.append(node)
    while queue:
        m=queue.pop(0)
        print(m,end=" ")
        for neigh in graph[m]:
            if neigh not in visited:
                visited.append(neigh)
                queue.append(neigh)
print(bfs(visited,graph,'S'))


# In[ ]:


##practical 5 depth first search


# In[5]:


#DFS#
graph={'S':['A','B','C'],
          'A':['S','D'],
         'B':['S','E'],
         'C':['S','F'],
         'D':['A','G'],
         'E':['B','G','F'],
         'F':['C','E'],
          'G':['D','E']}
visited=[]
def dfs(visited,graph,node):
    if node not in visited:
        print(node)
        visited.append(node)
        for neigh in graph[node]:
            dfs(visited,graph,neigh)
print(dfs(visited,graph,'S'))


# In[ ]:


##practical 6 A* algorithm


# In[7]:


#A star
nodeList={'mumbai':[("delhi",1200),("nasik",350),("goa",800),("pune",130)],
        "delhi":[("nasik",375),("mumbai",1200)],
        "nasik":[("indore",600),("delhi",375),("mumbai",350),("nagpur",600)]
,
        "indore":[("nasik",400)],
        "nagpur":[("nasik",600),("pune",450)],
        "pune":[("mumbai",130),("nagpur",450),("blore",550)],
        "blore":[("hyd",110),("goa",750)],
        "goa":[("blore",750),("hyd",850),("mumbai",800)],
        "hyd":[("blore",750),("goa",750)]}
hd={"mumbai":790,"delhi":1515,"nasik":1140,"indore":1540,"nagpur":1110,"pune":
660,"blore":110,"goa":850,"hyd":0}
openList=[("mumbai",700)]
closedList=[]
def goalTest(node):
    return node=="hyd"
def moveGen(node):
    return nodeList[node[0]]
def sort(mylist):
    for i in range(len(mylist)):
        for j in range(0,len(mylist)-i-1):
            if (mylist[j][1]>mylist[j+1][1]):
                temp=mylist[j]
                mylist[j]=mylist[j+1]
                mylist[j+1]=temp
    return (mylist)
def AStar():
    while(len(openList)>0):
        sort(openList)
        print("Open List Contains:",openList)
        node=openList.pop(0)
        closedList.append((node[0],hd[node[0]]))
        print("Picked node:",node)
        if (goalTest(node[0])):
            return "Goal Found"
        else:
            neighbours=moveGen(node)
            print("Neighbours of ",node," are: ",neighbours)
            for node in neighbours:
                if node not in openList and node[0] not in closedList[0]:
                    tup=(node[0],(node[1]+hd[node[0]]))
                    openList.append(tup)
        return "Goal Not Found"
AStar()
'''
Goal='hyd'
finding hd->>selecting mumbai
hd1=mumbai->delhi(1200)->nasik(375)->indore(600)->nasik(375)loop hence 
discarded
hd2=mumbai->nasik(350)->discarded
hd3=mumbai->goa(800)->blore(750)->hyd
hd4=
hd5=
hd6=
'''


# In[ ]:


##practical 7 best first search


# In[8]:


#BEST FIRST SEARCH
map_list={'Mumbai':[('Pune',750),('Delhi',1500),('Goa',1300)],
        'Goa':[('Mumbai',1200)],
        'Delhi':[('Mumbai',1200),('Guwahati',100),('Pune',750)],
        'Chennai':[('Pune',750)],
        'Kolkata':[('Guwahati',100),('Pune',750)],
        'Pune':[('Mumbai',1200),('Kolkata',0),('Chennai',1600),('Delhi',1500)],
        'Guwahati':[('Delhi',1500),('Kolkata',0)]
            }
OPEN=[[('Mumbai',1200),None]]
CLOSED=[]
def movegen(node):
    return map_list[node]
def goaltest(node):
    return node=='Kolkata'
final=[]
def reconstructpath(path):
    if path is None:
        return ""
    else:
        final.append(path[0][0])
        reconstructpath(path[1])
        return final
def sort(a):
    for i in range(len(a)):
        for j in range(0,len(a)-i-1):
            if ((a[j][0][1]),a[j+1][0][1]):
                (a[j],a[j+1])=(a[j+1],a[j])
    return a
def best():
    while len(OPEN)>0:
        print("Open List:",OPEN)
        x=sort(OPEN)
        seen=x.pop(0)
        N=seen[0][0]
        CLOSED.append(N)
        print("Closed List cotains",CLOSED)
        print("Node Picked:",N)
        if goaltest(N):
            print(reconstructpath(seen)[::-1])
            return "Found"
        else:
            neigh=movegen(N)
        for i in neigh:
            if i[0] not in CLOSED and i not in OPEN:
                new=[i,seen]
                OPEN.append(new)
    return "Not Found"
best()


# In[ ]:


##practical 8 decision tree
##take sample data and balance scale data


# In[9]:


#decision tree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
balance_data=pd.read_csv("balance-scale.data",sep=",",header=None)
print("Dataset Length:: ",len(balance_data))
print("Dataset Shape:: ",balance_data.shape)
print(balance_data.head())
X=balance_data.values[:, 1:5]#predicator
Y=balance_data.values[:,0]#target attr (B,R,L)
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=100)
clf_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3,min_samples_leaf=5)
clf_entropy.fit(X_train,y_train)
clf_gini=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=3,min_samples_leaf=5)
clf_gini.fit(X_train,y_train)
print(y_test)
y_pred_en=clf_entropy.predict(X_test)
y_pred_gini=clf_gini.predict(X_test)
print(y_pred_en)
print(y_pred_gini)
a=accuracy_score(y_pred_en,y_test)*100
b=accuracy_score(y_pred_gini,y_test)*100
print(a)
print(b)


# In[ ]:


##practical 9 svc classification
## take sample data and balance scale data


# In[ ]:


#svc classification
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import datasets
balance_data=pd.read_csv('balance-scale.data',sep=",",header=None)
print("Dataset Length:: ",len(balance_data))
print("Dataset Shape:: ",balance_data.shape)
print(balance_data.head())
X=balance_data.values[:,1:5]
Y=balance_data.values[:,0]
X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.3,random_state=100)
svclassifier=SVC(kernel="linear")
svclassifier.fit(X_train,y_train)
print(y_test)
y_pred=svclassifier.predict(X_test)
print(y_pred)
print(accuracy_score(y_pred,y_test)*100)


# In[ ]:


##practical 10 adaboost classification
##take sample data and balance-scale data


# In[10]:


#AdaBoostClassification
import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names= ['preg','plas','pres', 'skin', 'test','mass', 'pedi','age', 'class']
dataframe = pandas.read_csv(url, names=names)
array=dataframe.values
print("Dataset Length:: ", len(dataframe))
print("Dataset Shape:: ", dataframe.shape)
print(dataframe.head())
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees=30
kfold = model_selection.KFold(n_splits=10)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean()*100)


# In[ ]:




