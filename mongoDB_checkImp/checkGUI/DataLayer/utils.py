# 本节提供一些基本的操作
import simplejson as json
import os
import random
import numpy as np
import queue

random.seed(42)
def readJsonFile(filePath):
    with open(filePath, 'r') as f:
        data = json.load(f)
    return data

def writeJsonFile(filePath, data):
    with open(filePath, 'w') as f:
        json.dump(data, f)

def symmetric_matrix_addition(matrix):
    """
    将给定的numpy方阵的上三角和下三角进行对称相加。

    参数:
    matrix (numpy.ndarray): 输入的方阵

    返回:
    numpy.ndarray: 对称相加后的矩阵
    """
    # 获取矩阵的维度
    n, m = matrix.shape

    # 检查是否为方阵
    assert n == m, "输入必须为方阵"
    # 创建一个新的矩阵,用于存储对称相加的结果
    result = np.zeros_like(matrix)
    # 遍历矩阵,进行对称相加
    for i in range(n):
        for j in range(i, n):
            result[i, j] = matrix[i, j] + matrix[j, i]
            result[j, i] = result[i, j]
    return result

# 写一个随机选择的函数

def random_select(ls, ratio):
    """
    从给定的列表中随机选择指定比例的元素。

    参数:
    ls (list): 输入的列表
    ratio (float): 选择的比例

    返回:
    list: 随机选择的元素列表
    """
    # 计算选择的数量
    num = int(len(ls) * ratio)
    # 从列表中随机选择元素
    result = random.sample(ls, num)
    return result
