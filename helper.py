import numpy as np

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

def print_matrix(matrix, name):
    print(f"This dude ({name}) looks like this: ")
    print(matrix)

def softmax(x):
    x = np.array(x)
    # subtract max for numerical stability and calculate exponentials
    exp_x = np.exp(x - np.max(x))
    
    # normalize by dividing each by the sum
    return exp_x / np.sum(exp_x)

def rowmax(matrix):
    output = np.full((len(matrix), 1), 0)

    for i in range(len(matrix)):
        output[i][0] = max(matrix[i])

    return output

def rowsum(matrix):
    output = np.full((len(matrix), 1), 0)

    for i in range(len(matrix)):
        output[i][0] = sum(matrix[i])

    return output

def diag(tensor):
    output = np.full((len(tensor), len(tensor)), 0)

    for i in range(len(tensor)):
        output[i][i] = tensor[i][0]

    return output

def inverse(tensor):
    return np.linalg.inv(tensor)