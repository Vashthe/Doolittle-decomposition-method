import numpy as np


def lu_decomposition(A):
    """
    LU分解函数（带部分主元选择）
    参数:
        A: 待分解的n×n方阵
    返回:
        L: 单位下三角矩阵
        U: 上三角矩阵
    """
    n = A.shape[0]  # 矩阵维度
    L = np.eye(n)  # 初始化L为单位矩阵（对角线为1）
    U = A.copy()  # 初始化U为A的副本

    for k in range(n - 1):  # 对每列进行消元（n-1步）
        # 部分主元选择：找到当前列下方绝对值最大的行
        pivot_row = np.argmax(np.abs(U[k:, k])) + k

        # 行交换（提高数值稳定性）
        U[[k, pivot_row]] = U[[pivot_row, k]]

        # 如果k>0，需要同步交换L矩阵已计算的部分
        if k > 0:
            L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        # 高斯消元
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]  # 计算并存储消元系数
            U[i, k:] -= L[i, k] * U[k, k:]  # 消去下方行

    return L, U


def forward_substitution(L, b):
    """
    前向替换（解Ly = b）
    参数:
        L: 单位下三角矩阵
        b: 右侧向量
    返回:
        y: 中间解向量
    """
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        # y[i] = (b[i] - Σ L[i,j]*y[j]) / L[i,i]
        # 因为L是单位下三角矩阵（L[i,i]=1），所以简化为：
        y[i] = b[i] - np.dot(L[i, :i], y[:i])  # 向量点积优化计算
    return y


def backward_substitution(U, y):
    """
    回代（解Ux = y）
    参数:
        U: 上三角矩阵
        y: 前向替换得到的向量
    返回:
        x: 最终解向量
    """
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):  # 从最后一行倒序计算
        # x[i] = (y[i] - Σ U[i,j]*x[j]) / U[i,i]
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def solve_lu(A, b):
    """
    使用LU分解求解Ax = b
    参数:
        A: 系数矩阵
        b: 右侧向量
    返回:
        x: 解向量
    """
    # 1. LU分解
    L, U = lu_decomposition(A)
    print("L矩阵为：")
    print(L)
    print("-"*50)
    print("U矩阵为：")
    print(U)
    print("-"*50)

    # 2. 前向替换解 Ly = b
    y = forward_substitution(L, b.flatten())  # 确保b是一维向量

    # 3. 回代解 Ux = y
    x = backward_substitution(U, y)

    return x


# 测试用例
if __name__ == "__main__":
    # 输入矩阵和向量
    A = np.array([[1, 2, 3],
                  [1, 3, 5],
                  [1, 3, 6]], dtype=float)
    b =  np.array([2, 4, 5],  dtype=float)

    # 求解并打印结果
    x = solve_lu(A, b)
    print("解向量 x:")
    print(x)
    # 应输出 [1. 1. -1.]
