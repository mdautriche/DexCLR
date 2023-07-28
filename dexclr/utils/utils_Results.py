import pandas as pd
import matplotlib.pyplot as plt

def best_lr(matrix_IOU):
    return matrix_IOU.IOU[matrix_IOU.IOU == max(
        matrix_IOU.IOU)].index[0]

def read_Matrix(filename):
    return pd.read_csv(
        filename, index_col=0).set_index('LR')
