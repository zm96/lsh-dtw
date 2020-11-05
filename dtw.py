import numpy as np
import pandas as pd
import time 
from tslearn.metrics import dtw, dtw_path
# 读取database，去掉第一列，转化为array
database = pd.read_csv('D:/workspace/DTW/Plane/Plane_TEST.tsv', sep='\t', header=None).drop([0], axis=1).values
query = pd.read_csv('D:/workspace/DTW/Plane/Plane_TRAIN.tsv', sep='\t',header=None).drop([0], axis=1).values
query = query[:90]
threshold = 10.0  # 相似度阈值
rowNum = 3
colNum = 44  # 桶的个数 
m = 3  

def divCell(data):
    seqID = np.zeros(data.shape, dtype = int)
    for i in range(len(data)):
        mean = np.mean(data[i])
        std = np.std(data[i])
        data[i] = (data[i]-mean)/std
        dmax = data[i].max()
        dmin = data[i].min()
        length = np.size(data,1)
        cellLength = (dmax-dmin)/rowNum*1.01
        cellWidth = (length-1)/colNum*1.01
        for j in range(length):
            row = np.floor((data[i][j]-dmin)/cellLength)+1
            col = np.floor(j/cellWidth)+1
            if col>colNum:
                col = colNum
            seqID[i][j]=(row-1)*colNum+col
    return seqID

def Embedding(seqID,seed):
    length = np.size(seqID,1)
    N = 3*length
    np.random.seed(seed)
    rand = np.random.randint(0,2,N)
    embedID = np.zeros([len(seqID),N], dtype = int)
    for i in range(len(seqID)):
        j = 0
        k = 0
        while(j<N):
            if k<length:
                embedID[i][j]=seqID[i][k]
                k += rand[j]
            else:
                embedID[i][j]=0
            j += 1
    return embedID

def Lsh(embedID,m,seed):
    length = np.size(embedID,1)
    np.random.seed(seed)
    hashPara = np.random.choice(length, size=m, replace=False, p=None)
    hashBucket = np.zeros([len(embedID),m], dtype = int)
    for i in range(len(embedID)):
        for j in range(m):
            hashBucket[i][j] = embedID[i][hashPara[j]]
    return hashBucket

time_start = time.time()
pointID = divCell(database)
seqID = divCell(query)
# 进行多次哈希
hashbucket1 = Lsh(pointID,m,1)
hashBucket1 = Lsh(seqID,m,1)
hashbucket2 = Lsh(pointID,m,2)
hashBucket2 = Lsh(seqID,m,2)
hashbucket3 = Lsh(pointID,m,3)
hashBucket3 = Lsh(seqID,m,3)
hashbucket4 = Lsh(pointID,m,4)
hashBucket4 = Lsh(seqID,m,4)
hashbucket5 = Lsh(pointID,m,5)
hashBucket5 = Lsh(seqID,m,5)
hashbucket6 = Lsh(pointID,m,6)
hashBucket6 = Lsh(seqID,m,6)
hashbucket7 = Lsh(pointID,m,7)
hashBucket7 = Lsh(seqID,m,7)
hashbucket8 = Lsh(pointID,m,8)
hashBucket8 = Lsh(seqID,m,8)
hashbucket9 = Lsh(pointID,m,9)
hashBucket9 = Lsh(seqID,m,9)
resultlist = []
for i in range(len(hashBucket1)):   # 遍历查询序列
    value = ""    
    for j in range(len(hashbucket1)):  # 遍历数据库序列
        flag1 = (hashBucket1[i]==hashbucket1[j]).all() # 哈希族是否相等
        flag2 = (hashBucket2[i]==hashbucket2[j]).all()
        flag3 = (hashBucket3[i]==hashbucket3[j]).all()
        flag4 = (hashBucket4[i]==hashbucket4[j]).all()
        flag5 = (hashBucket5[i]==hashbucket5[j]).all()
        flag6 = (hashBucket6[i]==hashbucket6[j]).all()
        flag7 = (hashBucket7[i]==hashbucket7[j]).all()
        flag8 = (hashBucket8[i]==hashbucket8[j]).all()
        flag9 = (hashBucket9[i]==hashbucket9[j]).all()
        if flag1 or flag2 or flag3 or flag4 or flag5 or flag6 or flag7 or flag8 or flag9: #只要有哈希族匹配就计算dtw
            optimal_path, dtw_score = dtw_path(query[i], database[j])
            if dtw_score <= threshold:
                value += str(j)+" "
    resultlist.append(value)
time_end = time.time()
time_avg = (time_end - time_start)/len(query)
print('LSH_DTW: ',time_avg)

time_start = time.time()
resultlist_true = []
for i in range(len(query)):
    value = ""    
    for j in range(len(database)):
        optimal_path, dtw_score = dtw_path(query[i], database[j])
        if dtw_score <= threshold:
            value += str(j)+" "
    resultlist_true.append(value)
time_end = time.time()
time_avg = (time_end - time_start)/len(query)
print('DTW: ',time_avg)

# 计算recall
r_list = [] 
for i in range(len(resultlist_true)):
    pre = set(resultlist[i].split( ))
    true = set(resultlist_true[i].split( ))
    r = len(pre&true)/len(true)
    r_list.append(r)
print('recall: ',np.mean(r_list))

