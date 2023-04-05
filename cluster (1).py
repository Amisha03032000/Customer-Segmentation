import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

#load the data
df=pd.read_csv('Mall_Customers.csv')

#find x
x=df.iloc[:,[3,4]].values
print(x)

#find the k(put randomly )
#k=6
#edited: add a method to calculate optimal k value
def find_optimal_k():
    wss=[]
    for index in range(1,11):
        kmeans=KMeans(n_clusters=index)
        kmeans.fit(x)
        wss.append(kmeans.inertia_)
    plt.scatter(range(1,11),wss,color='red')
    plt.plot(range(1,11),wss)
    plt.xlabel('K values')
    plt.ylabel('wss')
    plt.show()
find_optimal_k()





def evaluate():
    k=6
    from sklearn.cluster import KMeans
    kmeans=KMeans(n_clusters=k)
    clusters=kmeans.fit_predict(x)
    print(clusters) #which data belongs to which cluster

    print(kmeans.cluster_centers_) #gives all the clusters and their centroids

    print('-'*40)

    #visualize
    plt.scatter(x[clusters==0,0],x[clusters==0,1],color='red',label='Cluster0')
    plt.scatter(x[clusters==1,0],x[clusters==1,1],color='pink',label='Cluster1')
    plt.scatter(x[clusters==2,0],x[clusters==2,1],color='yellow',label='Cluster2')
    plt.scatter(x[clusters==3,0],x[clusters==3,1],color='cyan',label='Cluster3')
    plt.scatter(x[clusters==4,0],x[clusters==4,1],color='orange',label='Cluster4')
    plt.scatter(x[clusters==5,0],x[clusters==5,1],color='purple',label='Cluster5')


    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black',label='Centroid')


    plt.xlabel('Customer income')
    plt.ylabel('Customer spending score')
    #plt.tight_layout()
    plt.title('CLusters of customer')
    plt.legend()
    plt.show()

evaluate()