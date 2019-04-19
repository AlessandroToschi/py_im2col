import numpy as np

W = 640
H = 480

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


def conv2D(input_tensor, weights, output_tensor):
    pass

def main():
    input_tensor = np.random.rand(M, H, W)
    weights = np.random.rand(N, M, KH, KW)
    output_tensor = np.zeros((N, H, W))
    im2col(input_tensor)
    print(input_tensor.dtype)

if __name__ == "__main__":
    main()