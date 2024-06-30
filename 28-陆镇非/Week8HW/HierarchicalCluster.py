# Author: Zhenfei Lu
# Created Date: 5/24/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import numpy as np
import sys
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

class HNode(object):
    def __init__(self, data, index, distance):
        self.center = data
        self.index = index
        self.distance = distance
        self.children = []

    def getNodesTotalCountinOriginalData(self, N):  # be same as getting leaf node total count
        if (self is None):
            print("come 2 self is none part")
            return None
        totalCount = 0
        if(self.index < N):
            totalCount = 1  # self count included in
        for child in self.children:
            count = child.getNodesTotalCountinOriginalData(N)
            totalCount = totalCount + count
        return totalCount

    def getNodesTotalCount(self):
        if (self is None):
            print("come 2 self is none part")
            return None
        totalCount = 1  # self count included in
        for child in self.children:
            count = child.getNodesTotalCount()
            totalCount = totalCount + count
        return totalCount

    def getLeafNodesTotalCount(self):
        if (self is None):
            print("come 2 self is none part")
            return None
        totalCount = 0
        if(len(self.children) == 0):  # leaf node
            totalCount = 1
        for child in self.children:
            count = child.getLeafNodesTotalCount()
            totalCount = totalCount + count
        return totalCount

    def printTreeDFS(self):
        if (self is None):
            print("come 2 self is none part")
            return
        print(f"center: {self.center}, index: {self.index}, distance: {self.distance} ")
        for child in self.children:
            child.printTreeDFS()
        return

    def printTreeBFS(self):
        if (self is None):
            print("come 2 self is none part")
            return
        heap = []
        depth = 0
        depth_current = depth
        oneLayerNodesList = []
        heap.append((self, depth))
        while(len(heap) > 0):
            node, depth = heap.pop(0)
            if depth > depth_current:
                for node_ in oneLayerNodesList:
                    print(f"center: {node_.center}, index: {node_.index}, distance: {node_.distance} ", end="  ")
                print()
                depth_current = depth
                oneLayerNodesList = []
            oneLayerNodesList.append(node)
            for child in node.children:
                heap.append((child, depth+1))
        return

    def getAllLeafNodes(self):
        if (self is None):
            print("come 2 self is none part")
            return None
        leafNodeList = []
        if(len(self.children) == 0):
            leafNode = self
            leafNodeList.append(leafNode)
        for child in self.children:
            temp_leafNodeList = child.getAllLeafNodes()
            leafNodeList.extend(temp_leafNodeList)
        return leafNodeList

    def drawTreeBFS(self, fontsize=8):
        if (self is None):
            print("come 2 self is none part")
            return
        plt.figure()
        plt.title("TreeBFS")
        plt.axis('off')
        heap = []
        heap.append((self, 0, 0))  # (node, depth, x_start)
        while (len(heap) > 0):
            (node, depth, x_start) = heap.pop(0)
            leafNode_count = node.getLeafNodesTotalCount()
            x = x_start + leafNode_count / 2  # divide 2 for make the self-node in the middle of its child-nodes
            y = -depth
            plt.text(x, y, str(node.index), ha='center', va='center', bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle=f'round,pad={fontsize/80}'), fontsize=fontsize)
            # print(f"center: {node.center}, index: {node.index}, distance: {node.distance}  ")
            for child in node.children:
                heap.append((child, depth+1, x_start))
                leafNode_count = child.getLeafNodesTotalCount()
                x_child = x_start + leafNode_count / 2
                x_start = x_start + leafNode_count
                plt.plot([x, x_child], [y, -(depth+1)], "b-")  # draw line between node in layer_i and layer_i+1
        return

    def drawTreeDFS(self, x_start, depth, fontsize=8, isRoot=True):
        if (self is None):
            print("come 2 self is none part")
            return
        if(isRoot):
            plt.figure()
            plt.title("TreeDFS")
            plt.axis('off')

        leafNode_count = self.getLeafNodesTotalCount()
        x = x_start + leafNode_count / 2  # divide 2 for make the self-node in the middle of its child-nodes
        y = -depth
        plt.text(x, y, str(self.index), ha='center', va='center',
                 bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle=f'round,pad={fontsize / 80}'), fontsize=fontsize)
        # print(f"center: {node.center}, index: {node.index}, distance: {node.distance}  ")
        for child in self.children:
            child.drawTreeDFS(x_start, depth+1, fontsize=8, isRoot=False)
            leafNode_count = child.getLeafNodesTotalCount()
            x_child = x_start + leafNode_count / 2
            x_start = x_start + leafNode_count
            plt.plot([x, x_child], [y, -(depth + 1)], "d-")  # draw line between node in layer_i and layer_i+1
        return


class HierarchicalCluster(object):
    def __init__(self):
        pass

    def fit(self, data: np.ndarray):
        dist_dict = {}
        N = data.shape[0]  # original dataSet number N
        h_node_list = []
        for i in range(0, N):  # init node list
            h_node = HNode(data[i], i, None)
            h_node_list.append(h_node)
        # print(h_node_list)

        for i in range(0, N):  # calculate all distance, and record in a dict[tuple(index1, index2), distance]
            for j in range(i+1, N):
                dist = np.linalg.norm(h_node_list[i].center - h_node_list[j].center, axis=0, ord=2)
                keytuple = (i, j)
                dist_dict[keytuple] = dist

        mergeHistory_list = []
        usedIndex = set()
        for i in range(0, sys.maxsize):
            min_key = min(dist_dict, key=dist_dict.get)
            # print(min_key)
            new_center = (h_node_list[min_key[0]].center + h_node_list[min_key[1]].center) / 2
            h_node = HNode(new_center, N+i, dist_dict[min_key])
            h_node.children.append(h_node_list[min_key[0]])  # left node
            h_node.children.append(h_node_list[min_key[1]])  # right node
            h_node_list.append(h_node)
            record = [min_key[0], min_key[1], dist_dict[min_key], h_node.getLeafNodesTotalCount()]
            mergeHistory_list.append(record)
            usedIndex.add(min_key[0])
            usedIndex.add(min_key[1])
            # print(record)
            if(record[3] >= N):
                print("Solved: one of the clusters total number is equal to the total number!")
                break

            dist_dict = {}  # clear
            for m in range(0, N+i+1):  # calculate all distance again for the next round of loop, but except merged indexes
                if(m in usedIndex):
                    continue
                for n in range(m + 1, N+i+1):
                    if (n in usedIndex):
                        continue
                    dist = np.linalg.norm(h_node_list[m].center - h_node_list[n].center, axis=0, ord=2)
                    keytuple = (m, n)
                    dist_dict[keytuple] = dist
        return (np.array(mergeHistory_list), h_node_list[-1], h_node_list)


    def plotDendrogramByThirdPartyLib(self, hist):  # third party binary tree plot
        # scipy.cluster.hierarchy existed libs:
        # Z = linkage(X, 'ward')
        # f = fcluster(Z, 4, 'distance')
        plt.figure()
        plt.title('dendrogram')
        plt.xlabel('index')
        plt.ylabel('distance')
        dn = dendrogram(hist)

    def getTypesBydistance(self, distance_threshold, hist, data):
        N = data.shape[0]
        dict_index_typeCenter = {}
        for i in range(N):
            dict_index_typeCenter[i] = [-1, data[i]]  # init index-[type, center] dict
        N_hist = hist.shape[0]

        h_node_list = []
        for i in range(0, N):  # init node list
            h_node = HNode(data[i], i, None)
            h_node_list.append(h_node)

        typeLabel = 0
        for i in range(0, N_hist):
            if(hist[i, 2] > distance_threshold):
                break
            index1 = int(hist[i, 0])
            index2 = int(hist[i, 1])
            new_center = (h_node_list[index1].center + h_node_list[index2].center) / 2
            h_node = HNode(new_center, N+i, hist[i, 2])
            h_node.children.append(h_node_list[index1])  # left node
            h_node.children.append(h_node_list[index2])  # right node
            h_node_list.append(h_node)

            leafNodeList = h_node_list[-1].getAllLeafNodes()
            merged_indexes = [leafNode.index for leafNode in leafNodeList]
            # print(merged_indexes)
            for merged_index in merged_indexes:
                dict_index_typeCenter[merged_index] = [typeLabel, h_node_list[-1].center]
            typeLabel = typeLabel + 1

        typeLabel_current = typeLabel
        for k, v in dict_index_typeCenter.items():  # for the rest of individual data point (seperately point) which has not been merged
            if(v[0] == -1):
                dict_index_typeCenter[k][0] = typeLabel_current
                typeLabel_current = typeLabel_current + 1
        # because the typeLabel here may not from 0 start, need mapping back to 0 start again
        # print(dict_index_typeCenter)
        min_typeLabel = min([dict_index_typeCenter[k][0] for k in dict_index_typeCenter.keys()])
        # print(min_typeLabel)
        delta = min_typeLabel - 0
        for k, v in dict_index_typeCenter.items():
            dict_index_typeCenter[k][0] = v[0] - delta
        # print(dict_index_typeCenter)
        return dict_index_typeCenter

    def getTypesBydistanceMethod2(self, distance_threshold, hist ):
        def getIndexInOriginalDataList(indexList, N):
            outputIndexList = []
            for index in indexList:
                if index < N:
                    outputIndexList.append(index)
            return outputIndexList

        N = hist.shape[0] + 1
        dict_index_type = {}
        for i in range(N):
            dict_index_type[i] = -1  # init index-type dict
        N_hist = hist.shape[0]

        node_index_list = []
        for i in range(N):
            node_index_list.append([])

        typeLabel = 0
        for i in range(0, N_hist):
            if (hist[i, 2] > distance_threshold):
                break
            index1 = int(hist[i, 0])
            index2 = int(hist[i, 1])
            temp_index_list = [index1, index2]
            temp_index_list.extend(node_index_list[index1])
            temp_index_list.extend(node_index_list[index2])
            node_index_list.append(temp_index_list)

            merged_indexes = getIndexInOriginalDataList(node_index_list[-1], N)
            # print(merged_indexes)
            for merged_index in merged_indexes:
                dict_index_type[merged_index] = typeLabel
            typeLabel = typeLabel + 1

        # # do not need to do the second loop again below for picking the new merged node in a separate list and giving type label. It's slow for O(2*N). Do two times. and Waste time.
        # none_originalData_list = node_index_list[N:len(node_index_list)]  # no original data in this list . index >= N
        # # print(len(none_leafNode_list))
        # for typeLabel, temp_list in enumerate(none_originalData_list, start=0):
        #     merged_indexes = getIndexInOriginalDataList(temp_list, N)
        #     # print(merged_indexes)
        #     for merged_index in merged_indexes:
        #         dict_index_type[merged_index] = typeLabel

        typeLabel_current = typeLabel
        for k, v in dict_index_type.items():  # for the rest of individual data point (seperately point) which has not been merged
            if (v == -1):
                dict_index_type[k] = typeLabel_current
                typeLabel_current = typeLabel_current + 1
        # because the typeLabel here may not from 0 start, need mapping back to 0 start again
        min_typeLabel = min([dict_index_type[k] for k in dict_index_type.keys()])
        # print(min_typeLabel)
        delta = min_typeLabel - 0
        for k, v in dict_index_type.items():
            dict_index_type[k] = v - delta
        # print(dict_index_type)
        return dict_index_type


    def fitMethod2(self, data: np.ndarray):
        def getIndexInOriginalDataList(indexList, N):
            outputIndexList = []
            for index in indexList:
                if index < N:
                    outputIndexList.append(index)
            return outputIndexList

        dist_dict = {}
        N = data.shape[0]  # original dataSet number N
        node_index_list = []
        node_data_list = []
        for i in range(0, N):  # init node list
            node_index_list.append([])
            node_data_list.append(data[i])
        # print(node_index_list)

        for i in range(0, N):  # calculate all distance, and record in a dict[tuple(index1, index2), distance]
            for j in range(i+1, N):
                dist = np.linalg.norm(node_data_list[i] - node_data_list[j], axis=0, ord=2)
                keytuple = (i, j)
                dist_dict[keytuple] = dist

        mergeHistory_list = []
        usedIndex = set()
        for i in range(0, sys.maxsize):
            min_key = min(dist_dict, key=dist_dict.get)
            # print(min_key)
            new_center = (node_data_list[min_key[0]] + node_data_list[min_key[1]]) / 2
            temp_index_list = [min_key[0], min_key[1]]
            temp_index_list.extend(node_index_list[min_key[0]])
            temp_index_list.extend(node_index_list[min_key[1]])
            node_index_list.append(temp_index_list)
            node_data_list.append(new_center)
            record = [min_key[0], min_key[1], dist_dict[min_key], len(getIndexInOriginalDataList(node_index_list[-1], N))]
            mergeHistory_list.append(record)
            usedIndex.add(min_key[0])
            usedIndex.add(min_key[1])
            # print(record)
            if(record[3] >= N):
                print("one of the clusters total number is equal to the total number")
                break

            dist_dict = {}  # clear
            for m in range(0, N+i+1):  # calculate all distance again for the next round of loop, but except merged indexes
                if(m in usedIndex):
                    continue
                for n in range(m + 1, N+i+1):
                    if (n in usedIndex):
                        continue
                    dist = np.linalg.norm(node_data_list[m] - node_data_list[n], axis=0, ord=2)
                    keytuple = (m, n)
                    dist_dict[keytuple] = dist
        # print(node_index_list)
        # print(mergeHistory_list)
        return np.array(mergeHistory_list)
