from sympy import *
import matplotlib.pyplot as plt
import numpy as np
import math

def function_L(X_L): # 返回函数值列表
    n = len(X_L)
    Y_L = []
    for i in range(n):
        value = (sin(X_L[i] / 180*pi) / math.log(X_L[i])) * cos(X_L[i] / 180*pi)
        value = value.evalf()
        Y_L.append(value)
    return Y_L

def function_R(X_L):
    n = len(X_L)
    Y_L = []
    for i in range(n):
        value = 1 / X_L[i]**2 + 1
        Y_L.append(value)
    return Y_L

def Lagrange(X_L, Y_L, x): # 计算拉格朗日插值
    n = len(X_L)
    result = 0.0
    for i in range(n):
        temp = Y_L[i]
        for j in range(n):
            if i != j:
                temp *= (x - X_L[j]) / (X_L[i] - X_L[j]) 
        result += temp
    return result

def Quotient(X_L): # 计算n阶差商
    i = 0
    Quotient = [0 for i in range(len(X_L))]
    Y_L = function_L(X_L)
    # Y_L = function_R(X_L)
    while i < len(X_L) - 1:
        j = len(X_L) - 1
        while j > i:
            if i == 0:
                Quotient [j] = (Y_L[j] - Y_L[j-1]) / (X_L[j] - X_L[j-1])
            else:
                Quotient [j] = (Quotient[j] - Quotient[j-1]) / (X_L[j] - X_L[j-i-1])
            j -= 1
        i += 1
    Quotient[0] = Y_L[0]
    return Quotient

def Newton(X_L, quotient, x): # 计算牛顿插值
    i = j = 0
    Polynomial = result = 0
    Coefficient = 1
    for i in range(len(quotient)):
        for j in range(i):
            Coefficient *= (x-X_L[j])
        if i != 0:
            Polynomial += Coefficient * quotient[i]
        Coefficient = 1
    result = quotient[0] + Polynomial
    return result

def Plot_Image(X_L, Y_L, X_C, Y_C, model): # 绘制图像， 模式1为拉格朗日插值法，模式2为牛顿插值法
    if model == 1:
        plt.title("Lagrange_interpolation")
    if model == 2:
        plt.title("Newton_interpolation")
    plt.plot(X_L, Y_L, 's', label = 'Original Values') # 方块点，标签为初始值
    plt.plot(X_C, Y_C, 'r', label = 'Interpolation Values') # 红色拟合线， 标签为插值值
    plt.X_laber = ('X')
    plt.Y_laber = ('Y')
    plt.legend(loc = 3) # 放plot的laber位置
    plt.show()


x_list = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 175, 180]
x = 50
num = 50

y_list = function_L(x_list)
yR_list = function_R(x_list)
q_list = Quotient(x_list)

x_c = np.linspace(np.min(x_list), np.max(x_list), endpoint = True)
y_c = []
for i in x_c:
    y_c.append(Lagrange(x_list, y_list, i))

Lagrange_result = Lagrange(x_list, y_list, x)
Plot_Image(x_list, y_list, x_c, y_c, 1)
print(Lagrange_result)

Newton_result = Newton(x_list, q_list, x)
Plot_Image(x_list, y_list, x_c, y_c, 2)
print(Newton_result)

Lagrange_Runge = Lagrange(x_list, yR_list, x)
print(Lagrange_Runge)

