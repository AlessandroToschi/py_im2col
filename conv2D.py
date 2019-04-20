import numpy as np
import time

W = 1920
H = 1080

KW = 3
KH = 3

M = 4
N = 10

def im2col(input_tensor):
    padded_input_tensor = np.zeros((M, H + KH - 1, W + KW - 1))
    padded_input_tensor[:, 1 : -1, 1 : -1] = input_tensor
    col_input_tensor = np.zeros((M, KW * KH, W * H))
    for y in range(H):
        for x in range(W):
            col_input_tensor[:, :, y * W + x] = padded_input_tensor[:, y : y + KH, x : x + KW].reshape(M, KW * KH)
    return col_input_tensor


def conv2D(col_input_tensor, weights, output_tensor):
    for n in range(N):
        kernel_weights = weights[n, :, :, :].reshape((M, 1, KW * KH))
        output_tensor[n, :, :] = np.sum(np.matmul(kernel_weights, col_input_tensor), axis=0).reshape((H, W))

def main():
    memory_size = 0

    input_tensor = np.random.rand(M, H, W)
    weights = np.random.rand(N, M, KH, KW)
    output_tensor = np.zeros((N, H, W))

    start = time.time()

    col_input_tensor = im2col(input_tensor)
    conv2D(col_input_tensor, weights, output_tensor)

    end = time.time()

    print("Execution time: {}".format(np.round(end - start, decimals=3)))

    memory_size += input_tensor.size * input_tensor.dtype.itemsize
    memory_size += weights.size * input_tensor.dtype.itemsize
    memory_size += output_tensor.size * input_tensor.dtype.itemsize
    memory_size += col_input_tensor.size * col_input_tensor.dtype.itemsize
    print(memory_size / (2 ** 20))


if __name__ == "__main__":
    main()