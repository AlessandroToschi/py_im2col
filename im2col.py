import numpy as np 
import time
from functools import partial

W = 1920
H = 1080
CHANNELS = 3
KW = 3
KH = 3

def dummy_area_with_if(image, weights):
    area = np.zeros((H, W))
    for y in range(H):
        for x in range(W):
            conv = np.zeros((1, CHANNELS))
            for dy in range(-1, KH - 1):
                for dx in range(-1, KW - 1):
                    if 0 <= x + dx < W and 0 <= y + dy < H:
                        conv += np.multiply(image[y + dy, x + dx, :], weights[dy + 1, dx + 1, :])
            area[y, x] = np.sum(conv)
    return area

def dummy_area_with_padding(padded_image, weights):
    area = np.zeros((H, W))
    for y in range(H):
        for x in range(W):
            #area[y, x] = np.sum(padded_image[y : y + KH, x : x + KW, :])
            conv = np.zeros((1, CHANNELS))
            for dy in range(KH):
                for dx in range(KW):
                    conv += np.multiply(padded_image[y + dy, x + dx, :], weights[dy, dx, :])
            area[y, x] = np.sum(conv)
    return area

def dummy_area_im2col(padded_image, weights):
    area = np.zeros((W * H, 1))
    for c in range(CHANNELS):
        feature_map = padded_image[:, :, c]
        im2col = np.zeros((W * H, KW * KH))
        for y in range(H):
            for x in range(W):
                im2col[y * W + x, :] = feature_map[y : y + KH, x : x + KW].reshape((1, KW * KH))
        area += np.dot(im2col, weights[:, :, c].reshape((KW * KH, 1)))
    return area.reshape((H, W))

def super_area_im2col(padded_image, weights):
    #area = np.zeros((W * H, 1))
    transposed_padded_image = np.transpose(padded_image, (2, 0, 1))
    transposed_weights = np.transpose(weights, (2, 0, 1)).reshape((CHANNELS, KW * KH, 1))
    im2col = np.zeros((CHANNELS, W * H, KW * KH))
    im2col2 = np.zeros((CHANNELS, W * H, KW * KH))
    for y in range(H):
        for x in range(W):
            im2col[:, y * W + x, :] = transposed_padded_image[:, y : y + KH, x : x + KW].reshape((CHANNELS, KW * KH))
    for c in range(CHANNELS):
        feature_map = padded_image[:, :, c]
        for y in range(H):
            for x in range(W):
                im2col2[c, y * W + x, :] = feature_map[y : y + KH, x : x + KW].reshape((1, KW * KH))
    assert np.array_equal(im2col, im2col2)
    #abba = np.sum(np.dot(im2col, transposed_weights), axis=0)
    #print(np.sum(abba, axis=2).shape)
    #return np.sum(abba, axis=0).reshape((H, W))
    #return np.sum(np.dot(im2col, transposed_weights), axis=().reshape((H, W))
    #return np.sum(np.dot(im2col, weights), axis=2).reshape((H, W))
    #for c in range(CHANNELS):
    #    feature_map = padded_image[:, :, c]
    #    im2col = np.zeros((W * H, KW * KH))
    #    for y in range(H):
    #        for x in range(W):
    #            im2col[y * W + x, :] = feature_map[y : y + KH, x : x + KW].reshape((1, KW * KH))
    #    area += np.dot(im2col, weights[:, :, c].reshape((KW * KH, 1)))
    #return area.reshape((H, W))

def measure_function(function, name):
    start = time.process_time()
    output = function()
    end = time.process_time()
    print("Elapsed time of {}: {} s".format(name, np.round(end - start, decimals=3)))
    return output

def main():
    padded_image = np.zeros((H + KH - 1, W + KW - 1, CHANNELS))
    padded_image[1 : -1, 1 : -1, :] = np.random.rand(H, W, CHANNELS)
    weights = np.random.rand(KH, KW, CHANNELS)

    #oracle = measure_function(partial(dummy_area_with_if, padded_image[1 : -1, 1 : -1, :], weights), "dummy_area_if")
    #output1 = measure_function(partial(dummy_area_with_padding, padded_image, weights), "dummy_area_with_padding")
    #output2 = measure_function(partial(dummy_area_im2col, padded_image, weights), "dummy_area_im2col")
    output3 = measure_function(partial(super_area_im2col, padded_image, weights), "super_area_im2col")

    #assert np.array_equal(oracle, output1)
    #assert np.array_equal(oracle, output2)
    assert np.array_equal(oracle, output3)

if __name__ == "__main__":
    main()