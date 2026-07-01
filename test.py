import numpy as np

def softmax(input):
    exp_list = np.array([np.exp(i) for i in input])
    sum_exp_list = sum(exp_list)
    return exp_list / sum_exp_list

x = [1, 2, 8]
print(softmax(x))