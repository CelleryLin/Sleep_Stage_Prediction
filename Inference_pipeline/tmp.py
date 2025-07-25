import numpy as np

def move_negative_to_bottom_ordered(matrix):

    arr = np.array(matrix)
    result = np.full_like(arr, -1)
    for col in range(arr.shape[1]):
        col_values = arr[:, col]
        non_neg_values = col_values[col_values != -1]
        result[:len(non_neg_values), col] = non_neg_values
    
    return result

# Example usage
matrix = [
    [1, 2, 3, -1, -1], 
    [-1, 4, 5, 6, -1], 
    [-1, -1, 7, 8, 9]
]

transformed_matrix = move_negative_to_bottom_ordered(matrix)
print(transformed_matrix)