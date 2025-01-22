import numpy as np

# A = np.ones((4,4))
# B = np.ones((4,4))
# C = np.ones((4,4))
A = np.array([[10,100,10,100],
              [1,20,2,40],
              [1,2,30,4],
              [10,6,70,8]], float)

B = np.array([[100,2,30,4],
              [5,6,70,8],
              [1,200,3,4],
              [50,6,7,80]], float)

C = np.array([[1,2,30,4],
              [5,60,7,8],
              [1,2,30,4],
              [50,60,7,80]], float)

def partition(tensor, dimensions):
    rowNum, colNum = dimensions
    tensor = np.array(tensor)

    # check if the dimensions are correct
    if tensor.shape[0] % rowNum != 0 or tensor.shape[1] % colNum != 0 or rowNum > tensor.shape[0] or colNum > tensor.shape[1]:
        print("!!! ERROR: the dimensions for the partition is not correct !!!")
        return None
        
    # calculate size of sub-matrices
    row_size = tensor.shape[0] // rowNum
    col_size = tensor.shape[1] // colNum
    
    result = []
    # iterate through partitions
    for i in range(rowNum):
        for j in range(colNum):
            # extract sub-matrix using numpy slicing
            sub_matrix = tensor[i*row_size:(i+1)*row_size, j*col_size:(j+1)*col_size]
            result.append(sub_matrix)
            
    return result

def rowmax(x):
    return np.max(x, axis=1, keepdims=True)


def rowsum(x):
    return np.sum(x, axis=1, keepdims=True)



print("The naive way of softmax")
def softmax(x):
    """
    Row-wise softmax of a 2D NumPy array.
    Equivalent to the lines:
        m = rowmax(x)
        e = np.exp(x - m)
        l = rowsum(e)
        return e / l
    """
    row_max = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - row_max)
    row_sum = np.sum(e, axis=1, keepdims=True)
    return e / row_sum

output = softmax(A@B)@C
print(output)


print("The distributed Way")

a1, a2, a3, a4 = partition(A, [2,2])
b1, b2, b3, b4 = partition(B, [2,2])
c1, c2, c3, c4 = partition(C, [2,2])

p1 = a1@b1 + a2@b3
p2 = a1@b2 + a2@b4
p3 = a3@b1 + a4@b3
p4 = a3@b2 + a4@b4

m1 = rowmax(p1)
e1 = np.exp(p1-m1)
l1 = rowsum(e1)

m2 = rowmax(p2)
e2 = np.exp(p2-m2)
l2 = rowsum(e2)

m3 = rowmax(p3)
e3 = np.exp(p3-m3)
l3 = rowsum(e3)

m4 = rowmax(p4)
e4 = np.exp(p4-m4)
l4 = rowsum(e4)

s11 = (e1@c1)/l1
s12 = (e1@c2)/l1
s21 = (e2@c3)/l2
s22 = (e2@c4)/l2
s33 = (e3@c1)/l3
s34 = (e3@c2)/l3
s43 = (e4@c3)/l4
s44 = (e4@c4)/l4

# M1 = np.max(m1, m2)
# L1 = 

