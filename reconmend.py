import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import time

def get_max(l):
    return max(l)

def initial_zero_matrix(r, c):
    return np.zeros((r,c))

def store_data(dataframe, matrix):
    n = 0.0
    for i in dataframe.itertuples():
        matrix[i[1]-1, i[2]-1] = i[3]
        n = n + 1.0
    return matrix,n

def load(rate, training, test, n, users):
    for i in range(0,users):
        temp = np.random.choice(rate[i,:].nonzero()[0], size = n, replace=False)
        training[i,temp] = 0.
        test[i,temp] = rate[i,temp]
    return training, test

def Cosine_Similarity(training):
    vectorProduct = np.dot(training, training.T)
    modOfVector = np.array([np.sqrt(np.diagonal(vectorProduct))])
    modProduct = modOfVector * modOfVector.T
    result = vectorProduct / modProduct
    return result

def Knn_reconmend_system(similarity,test,train,k):
    reconmend_Matrix = np.zeros(test.shape)
    for line in range(similarity.shape[0]):
        index = [np.argsort(similarity[:,line])[-2:-k-2:-1]]
        
        for member in range(train.shape[1]):
            fenzi = train[:,member][index].dot(similarity[:,line][index])
            fenmu = np.sum(similarity[:,line][index])
            reconmend_Matrix[line,member]=fenzi/fenmu
    print('Prediction based on top-' + str(k) + ' users similarity is done...')
    return reconmend_Matrix

def Mean_Square_erroe(reconmend_Matrix,test):
    Real_value = test[np.nonzero(test)].flatten()
    reconmend_value = reconmend_Matrix[np.nonzero(test)].flatten()
    MSE = np.average((Real_value-reconmend_value)**2)
    return MSE
# the dataset name
def implement_reconmend_system():
    star = time.time()
    filename = 'u.data'

    # read dataset
    dataframes = pd.read_csv(filename, sep = '\t', names = ['user', 'movie', 'rating', 'time'])

    # get the number of users and moives in the given dataset 
    numberOfUser = get_max(dataframes.user)
    numberOfMoive = get_max(dataframes.movie)

    # inital two zero matix
    rate = initial_zero_matrix(numberOfUser,numberOfMoive)
    test = initial_zero_matrix(numberOfUser, numberOfMoive)

    # store dataframe data into matrix and get the number of non-ZERO entries
    rate, numberOfnonZero = store_data(dataframes,rate)

    # training set
    training = rate.copy()

    # print the number of uses and movies 
    print("there are " + str(numberOfUser) +" users in this dataset")
    print("And " + str(numberOfMoive) +" movies in this dataset")

    # calculate the sparisity of matrix
    spars = numberOfnonZero / (numberOfUser * numberOfMoive)
    # 10% for test and 90% for trainzng
    percentage = 0.1
    numberOfTest = round(percentage * spars * numberOfMoive)

    # get the traing and test data
    training, test = load(rate, training, test, numberOfTest, numberOfUser)

    # calculate similarity matrix based on Cosine Similarity
    cos_simi = Cosine_Similarity(training)
    
    # CF, top-k user
    interval = range(5,200,10)
    result = []
    for i in interval:
        reconmend_Matrix = Knn_reconmend_system(cos_simi,test,training,i)
        mse = Mean_Square_erroe(reconmend_Matrix,test)
        result.append(mse)
    elapsed = (time.time()-star)
    y= np.arange(len(interval))
    plt.plot(y,result)
    plt.bar(y,result,alpha=0.4,align='center')
    for a,b in zip(y,result):
        plt.text(a, b+0.05, '%.5f' % b, ha='center', va= 'bottom',fontsize=7)
    plt.xticks(y,interval)
    plt.ylabel('Mean Square Error')
    plt.xlabel('Value of K')
    plt.title('Mean Square Error with different K finish in '+str(elapsed)+' seconds')
    plt.show()
    
    


implement_reconmend_system()



