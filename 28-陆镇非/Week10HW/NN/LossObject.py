# Author: Zhenfei Lu
# Created Date: 6/25/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import weakref

class LossObject(object):
    def __init__(self, graph, loss, yGroundTruth):
        self.graph = weakref.ref(graph)  # weakptr
        self.loss = loss
        self.yGroundTruth = yGroundTruth

    def bwd(self):
        self.graph().bwd(self.yGroundTruth)

    def __call__(self):
        return self.loss

