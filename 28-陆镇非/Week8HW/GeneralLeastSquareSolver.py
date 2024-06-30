# Author: Zhenfei Lu
# Created Date: 4/27/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

# this solver is generally solving for both non-linear and linear least square problems

import numpy as np
from enum import Enum
import sys

class LstSqrSolveMethod(Enum):
    NONE = 0
    PIV = 1
    SVD = 2
    DLS = 3
    GD = 4
    LINEAR_CASE = 5

class LossType(Enum):
    L2_NORM = 0
    L1_ABS = 1
    L1_ABS_VECROR = 2
    LOSS_NOT_CHANGE = 3

# solve general nonlinear problems:
# cost = residual.T * residual
# where residual = f(X, params) - Y.  X is input, params is optParams, Y is DesiredValue
# params is the optimization params that you want to solve.  be mandatory to input a initial guess
# X, Y are samples input(known) and groundTruth(DesiredValue)(known). both of them are not mandatory
# otherNoneDerivaiableParams may be other calculate params you need.  not mandatory
class GeneralLstSqrSolver(object):
    def __init__(self, residualFunc, derivaiableParams, X, Y, otherNoneDerivaiableParams): # derivaiableParams must be Nx1
        self.residualFunc = residualFunc
        self.derivaiableParams = derivaiableParams  # do not need deepcopy .copy, if it get a new object's address, it will not be able to modify the input object's inside value by the input object's address
        self.X = X
        self.Y = Y
        self.otherNoneDerivaiableParams = otherNoneDerivaiableParams
        self.derivaiableParamsNum = self.derivaiableParams.shape[0]
        # self.maxIter = 30
        self.epsilon = 0.0001
        self.lossRequest = None
        self.loss = None
        self.loss_previous = None
        self.solveMethod = LstSqrSolveMethod.NONE
        self.DLSDampingTerm = 0.01
        self.learningRate = 0.001  # if use Taylor expainsion, there's no stepSize or learning rate

    def getJacobian(self, residual):
        fwd_N = residual.shape[0]
        jaocbi = np.zeros((fwd_N, self.derivaiableParamsNum))
        for i in range(0, self.derivaiableParamsNum):
            derivaiableParamsDelta = np.zeros((self.derivaiableParamsNum, 1), self.derivaiableParams.dtype)
            derivaiableParamsDelta[i, 0] = self.epsilon
            residual1 = self.residualFunc(self.derivaiableParams + derivaiableParamsDelta, self.X, self.Y, self.otherNoneDerivaiableParams)
            # residual2 = self.residualFunc(self.derivaiableParams - derivaiableParamsDelta, self.X, self.Y, self.otherNoneDerivaiableParams)
            # jaocbi[:, i:(i+1)] = (residual1 - residual2) / (2*self.epsilon)
            jaocbi[:, i:(i+1)] = (residual1 - residual) / self.epsilon
        return jaocbi

    def getPInvBySVD(self, jacobi):
        U, sigma, VT = np.linalg.svd(jacobi)
        sigma_mat = np.zeros((U.shape[1], VT.shape[0]))
        for i in range(0, sigma.shape[0]):
            if (abs(sigma[i]) >= 1e-10):
                sigma_mat[i, i] = 1 / sigma[i]
        persudoInvTranspose = U @ sigma_mat @ VT
        persudoInv = persudoInvTranspose.T
        return persudoInv

    def bwd(self, residual):
        jacobi = self.getJacobian(residual)
        gradient = None
        if(self.solveMethod == LstSqrSolveMethod.DLS):
            gradient = np.linalg.inv(jacobi.T@jacobi + (self.DLSDampingTerm) * np.identity(self.derivaiableParamsNum)) @ jacobi.T @ residual
        elif (self.solveMethod == LstSqrSolveMethod.PIV):
            gradient = np.linalg.inv(jacobi.T @ jacobi) @ jacobi.T @ residual
        elif (self.solveMethod == LstSqrSolveMethod.SVD):
            gradient = self.getPInvBySVD(jacobi) @ residual
        elif (self.solveMethod == LstSqrSolveMethod.GD):  # NOT recommended
            gradient = self.learningRate * jacobi.T @ residual
        return (gradient, jacobi)

    def satisfiedLoss(self, residual):
        if(self.lossRequest[0] == LossType.L1_ABS_VECROR):
            self.loss = np.abs(residual)
            for i in range(0, self.loss.shape[0]):
                if(self.loss[i, 0] > self.lossRequest[1]):
                    return False
        elif(self.lossRequest[0] == LossType.L1_ABS):
            self.loss = np.sum(np.abs(residual), axis=0) / self.Y.shape[0]
            if (self.loss > self.lossRequest[1]):
                return False
        elif (self.lossRequest[0] == LossType.L2_NORM):
            self.loss = np.linalg.norm(residual, axis=0, ord=2) / self.Y.shape[0]
            if (self.loss > self.lossRequest[1]):
                return False
        elif (self.lossRequest[0] == LossType.LOSS_NOT_CHANGE):
            self.loss = residual
            for i in range(0, self.loss.shape[0]):
                if (abs(self.loss[i, 0] - self.loss_previous[i, 0]) > self.lossRequest[1]):
                    self.loss_previous = residual
                    return False
            self.loss_previous = residual
        return True

    def solve(self, maxIter: int, lossRequest: tuple, LstSqrSolveMethod = LstSqrSolveMethod.DLS):
        self.lossRequest = lossRequest
        self.solveMethod = LstSqrSolveMethod
        deltaMin = 0.01
        delta = deltaMin
        for i in range(0, maxIter):
            residual = self.residualFunc(self.derivaiableParams, self.X, self.Y, self.otherNoneDerivaiableParams)
            if(i == 0):
                if(self.lossRequest[0] == LossType.LOSS_NOT_CHANGE):
                    self.loss_previous = np.ones((residual.shape[0], 1)) * sys.maxsize
            if(self.satisfiedLoss(residual)):
                print('LstSqrSolver solved problem and satisfied Loss with iter: ' + str(i))
                return (True, self.derivaiableParams, self.loss)
            if(LstSqrSolveMethod != LstSqrSolveMethod.LINEAR_CASE):
                (gradient, jacobi) = self.bwd(residual)
                self.derivaiableParams = self.derivaiableParams - gradient
            else:
                A = self.getJacobian(residual)
                self.derivaiableParams = np.linalg.inv(A.T @ A) @ A.T @ self.Y  #  linear case: AX=b, X = pinv(A)*b
                residual = self.residualFunc(self.derivaiableParams, self.X, self.Y, self.otherNoneDerivaiableParams)
                self.satisfiedLoss(residual)
                print('LstSqrSolver solved linear case with iter: ' + str(i))
                return (True, self.derivaiableParams, self.loss)
        print('LstSqrSolver may not get a solution. Not satisfied Loss and reached the Max iter num: ' + str(maxIter))
        return (False, self.derivaiableParams, self.loss)


# trust region and dogleg
# J_k = jacobi
# F_k = residual
# # Solve the trust region subproblem
# p_k = -gradient
# # Compute the actual reduction
# F_new = self.residualFunc(self.derivaiableParams+p_k, self.X, self.Y, self.otherNoneDerivaiableParams)
# rho_k = (np.linalg.norm(F_k) ** 2 - np.linalg.norm(F_new) ** 2) / (
#             np.linalg.norm(F_k) ** 2 - np.linalg.norm(F_k + J_k @ p_k) ** 2)
# # rho_k = (np.linalg.norm(F_k)  - np.linalg.norm(F_new)) / (
# #         np.linalg.norm(F_k)- np.linalg.norm(F_k + J_k @ p_k) )
# print(rho_k)
# # Update x
# if rho_k > 0  :
#     self.derivaiableParams = self.derivaiableParams - gradient
#
# # Update trust region radius
# if rho_k <= 0.25:
#     delta *= 0.25
# elif rho_k > 0.75:
#     delta = min(2 * delta, 1)  # Limiting delta to 1.0 for stability
#
# # Update lambda
# if(rho_k > 0):
#     self.DLSDampingTerm = max(1e-7, self.DLSDampingTerm /2)
# else:
#     self.DLSDampingTerm = min(1e-7, self.DLSDampingTerm * 2)



# delta = 0.01
# min_delta = 1e-4
# g = jacobi.T @ residual
# B = jacobi.T @ jacobi
# # 解决信赖域子问题
# p_b = -np.linalg.inv(B + 0.01*np.eye(self.derivaiableParams.shape[0])).dot(g)  # 加入微小正则化项以确保矩阵可逆
# if np.linalg.norm(p_b) <= delta:
#     p = p_b
# else:
#     p = (delta / np.linalg.norm(p_b)) * p_b  # 缩放步长以满足信赖域约束
#
# # 计算 \(\rho\)
# new_loss = self.residualFunc(self.derivaiableParams + p, self.X, self.Y, self.otherNoneDerivaiableParams)
# actual_reduction = residual.T @ residual - new_loss.T @ new_loss
# predicted_reduction = -g.T  @(p) - 0.5 * p.T@(B)@(p)
# rho = actual_reduction / predicted_reduction
# print(rho)
# # 调整信赖域半径
# if rho < 0.25:
#     delta = max(0.25 * delta, min_delta)
# elif rho > 0.75 and np.linalg.norm(p) == delta:
#     delta = min(2 * delta, 1000.0)
#
# # 如果 \(\rho\) > 0，接受步长
# if rho > 0:
#     self.derivaiableParams = self.derivaiableParams + p



# new_params = self.derivaiableParams - gradient
# loss_new = self.residualFunc(new_params, self.X, self.Y, self.otherNoneDerivaiableParams)
# actual_reduction = residual.T@residual - loss_new.T@loss_new
# predicted_reduction = -0.5 * gradient.T @ jacobi.T@jacobi @ gradient - gradient.T @ jacobi.T@residual
# rho = actual_reduction / predicted_reduction
# print(rho)
# if rho > 0:
#     self.DLSDampingTerm = self.DLSDampingTerm * max(1/3, 1 - (2*rho - 1)**3)
# else:
#     self.DLSDampingTerm = 2 * self.DLSDampingTerm
# # if rho < 0.25:
# #     self.DLSDampingTerm = 0.25*self.DLSDampingTerm
# # elif rho > 0.75:
# #     self.DLSDampingTerm = 2 * self.DLSDampingTerm
#
# if np.abs(rho-0) <= 1e-5 or rho>0:
#     print("here")
#     self.derivaiableParams = self.derivaiableParams - gradient


