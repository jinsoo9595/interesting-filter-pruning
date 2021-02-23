
import numpy as np
from scipy.spatial import distance

def L1norm_method(weight):
    return np.linalg.norm(weight, 1, axis=1)

def L2norm_method(weight):
    return np.linalg.norm(weight, 2, axis=1)

def distance_cal_func(weight, distance_method):
    if distance_method == "euclidean":
        similar_matrix = distance.cdist(weight, weight, 'euclidean')
    elif distance_method == "mahalanobis":
        similar_matrix = distance.cdist(weight, weight, 'mahalanobis')

    return similar_matrix

# [Geometric Median with L1-norm process]
def geometric_median(weight, distance_method, norm_method):
    # To calculate distance between coordinates weights shape has to change

    # Example is 3D to 2D but its actually 4D to 2D
    # weight = array([[[ 7.,  6.,  7.],
    #                  [24., 27., 30.],
    #                  [51., 54., 57.]],
    #
    #                 [[24., 27., 30.],
    #                  [ 7.,  6.,  7.],
    #                  [51., 54., 57.]],
    #
    #                 [[ 7.,  6.,  7.],
    #                  [24., 27., 30.],
    #                  [51., 54., 57.]]], dtype=float32)

    ###########
    # Reduce dimension 4D -> 2D (to calculate distance)
    ###########
    # weight = array([[ 5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.],
    #                 [-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.],
    #                 [14., 15., 16., 17., 18., 19., 20., 21., 22.]], dtype=float32)
    # Weight : 3x3x3x64 => 27 x 64 => 64 x 27 (to rank each filter)
    weight = weight.reshape(-1, weight.shape[3])
    weight = np.transpose(weight)


    ###########
    # Normalization
    ###########
    # np.linalg.norm(input matrix,
    #                Order of the norm (see the link),
    #                specifies the axis of x along which to compute the vector norms)
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    weight_norm = _methods[norm_method](weight)
    # L1 => array([81., 20., 162.], dtype=float32)
    # L2 => array([28.089144,  7.745967, 54.552727], dtype=float32)


    ###########
    # Sort norm value => get index order
    ###########
    # get sorted filter index (int64)
    # [81., 20., 162.].argsort() => get index array [1, 0, 2]
    weight_sorted_index = weight_norm.argsort()


    ###########
    # Sort weight by index order
    ###########
    #                                                                                [order]
    # weight = array([[-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.],                    0
    #                 [ 5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.],                    1
    #                 [14., 15., 16., 17., 18., 19., 20., 21., 22.]], dtype=float32)    2
    # change matrix arrays in index order (torch.index_select)
    weight = weight[weight_sorted_index]


    ###########
    # Calculate distance between coordinates
    ###########
    # for euclidean/mahalanobis distance
    similar_matrix = distance_cal_func(weight, distance_method)


    ###########
    # Sum distance
    ###########
    similar_sum = np.sum(np.abs(similar_matrix), axis=0)

    return similar_sum

_methods = {
    'L1': L1norm_method,
    'L2': L2norm_method
}