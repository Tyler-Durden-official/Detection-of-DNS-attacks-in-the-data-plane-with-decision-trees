import numpy as np 
import pandas as pd
import csv
data=pd.read_csv('data_set.csv')
file=open("tokenised1.csv",'w+',newline='')

ll=[]

for iter,row in data.iterrows():
    
    l=[ord(c) for c in row['domain']]
    l=l[:20]
    length=len(l)
    if(length<20):
        for i in range(20-length):
            l.append(0)
    
    l.append(row['class'])
    ll.append(l)
    #count=count+1
    #if count==20:
    #    break

with file:
    writer=csv.writer(file)
    names=[i+1 for i in range(20)]
    names.append('class')
    writer.writerow(names)
    writer.writerows(ll)
