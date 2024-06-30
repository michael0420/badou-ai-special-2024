import numpy as np

class DObject(object):
    def __init__(self, index, data):
        self.index = None
        self.data = None


class DensityCluster(object):
    def __init__(self, eps, minPts):
        self.eps = eps
        self.minPts = minPts

    def fit(self, data):  # eps is radia distance, minPts is the count of min points
        N = data.shape[0]
        distanceMatrix = self.computeDistanceMatrix(data)  # could build KD tree here to speed up
        # print(distanceMatrix)
        dict_index_type = {}
        for i in range(0,N):
            dict_index_type[i] = -1   # init dict index-type , -1 means noise type

        typeLabel = 0
        merged_index_list = []  # index that has been used/merged
        DObjects = []  # except the noise type datas
        for i in range(N):  # choose a new data point as a cluster
            if(dict_index_type[i] != -1):  # if type != -1 means it has already been merged/clustered
                continue
            data_i_distance_list = distanceMatrix[i]
            desired_indexList = self.getIndexListSatisfiedEps(data_i_distance_list, i, merged_index_list)
            stack = []
            if(len(desired_indexList) >= self.minPts):
                dict_index_type[i] = typeLabel
                DObjects.append([])
                DObjects[-1].append(DObject(index=i, data=data[i]))
                merged_index_list.append(i)
                # find the desired point and add its desired point into a stack
                stack.extend(desired_indexList)
                merged_index_list.extend(desired_indexList)  # used index cannot be picked again
            else:
                continue

            while (len(stack) > 0):
                i = stack.pop()  # index = indexes pop.  Python language will not influence i value for the next round of loop if you asign i in the loop
                dict_index_type[i] = typeLabel
                DObjects[-1].append(DObject(index=i, data=data[i]))
                data_i_distance_list = distanceMatrix[i]
                desired_indexList = self.getIndexListSatisfiedEps(data_i_distance_list, i, merged_index_list)
                if (len(desired_indexList) >= self.minPts):
                    stack.extend(desired_indexList)
                    merged_index_list.extend(desired_indexList)  # used index cannot be picked again

            typeLabel = typeLabel + 1
        # print(dict_index_type)
        # print("DObjects")
        # print(len(DObjects))
        # print(merged_index_list)
        return dict_index_type, DObjects

    def computeDistanceMatrix(self, data):
        N = data.shape[0]
        distanceMatrix = np.zeros((N, N))
        for i in range(0, N):
            for j in range(0, N):
                distanceMatrix[i, j] = np.linalg.norm(data[i]-data[j], axis=0, ord=2)
        return distanceMatrix

    def getIndexListSatisfiedEps(self, distanceList, selfIndex, merged_index_list):
        N = distanceList.shape[0]
        desired_index_list = []
        for i in range(0, N):
            if(i == selfIndex):
                continue
            continue_flag = False
            for merged_index in merged_index_list:
                if(merged_index == i):
                    continue_flag = True
            if(continue_flag):
                continue
            if(distanceList[i] <= self.eps):
                desired_index_list.append(i)
        return desired_index_list
