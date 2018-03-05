__author__ = 'bohaohan'
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy import misc


x = np.loadtxt("small_test.csv", delimiter=",")  # load from a small part of test
print x.shape
x = x.reshape(-1, 64, 64)  # reshape
print x.shape


def threshold_(img, thres=250):
    img[img > thres] = 255
    img[img <= thres] = 0
    return img


def update_bound(i, j, bound):
    if i < bound[0]:
        bound[0] = i
    if i > bound[1]:
        bound[1] = i
    if j < bound[2]:
        bound[2] = j
    if j > bound[3]:
        bound[3] = j
    return bound


def find(x, i, j, record, bound=None):
    # s_h, e_h, s_w, e_w = 0, 0, 0, 0
    if i < 0 or i > 63 or j < 0 or j > 63:
        return bound

    if record[i][j] > 0.5 or x[i][j] < 240:
        return bound

    record[i][j] = 1

    if bound is None:
        bound = [0, 0, 0, 0]

    bound = update_bound(i, j, bound)

    step = [[1, 0], [0, -1], [-1, 0], [0, 1]]
    for step_ in step:
        find(x, i + step_[0], j + step_[1], record, bound)

    return bound


def detect(x):
    record = np.zeros(x.shape, dtype=np.int32)
    max_area = 0
    max_bound = None
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if record[i][j] > 0.5 or x[i][j] < 240:
                continue
            bound = [i, i, j, j]
            bound = find(x, i, j, record, bound)
            if bound is not None:
                cur_area = (bound[1] - bound[0]) * (bound[3] - bound[2])
                # cur_area = max(bound[1] - bound[0], bound[3] - bound[2])
                if cur_area > max_area:
                    max_area = cur_area
                    max_bound = bound

    # print bound
    padding = 2

    # padding the bound
    max_bound[0] = max_bound[0] - padding if max_bound[0] - padding > 0 else 0
    max_bound[1] = max_bound[1] + padding if max_bound[1] + padding < 64 else 64
    max_bound[2] = max_bound[2] - padding if max_bound[2] - padding > 0 else 0
    max_bound[3] = max_bound[3] + padding if max_bound[3] + padding < 64 else 64

    return x[max_bound[0]: max_bound[1], max_bound[2]: max_bound[3]]


def padding(image):
    img_size = (32, 32)
    h, w = image.shape
    pad_h = (max(h, w) - h) / 2
    pad_w = (max(h, w) - w) / 2
    new_image = np.zeros([h + 2 * pad_h, w + 2 * pad_w], dtype=np.float32)
    new_image[pad_h: pad_h + h, pad_w: pad_w + w] = image
    new_image = misc.imresize(new_image, img_size)
    return new_image


# def open_operation(image):
#     kernel = kernel = np.ones((3, 3), np.float32)
#     # erosion = cv2.erode(image, kernel, 1)
#     # erosion = cv2.dilate(erosion,kernel,1)
#     return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# x =

# visualization
for x_ in np.random.permutation(x):
# for x_ in x:
    plt.figure(0)

    sub = plt.subplot(2, 2, 1)
    sub.imshow(x_, cmap='gray')

    b_x = threshold_(copy.deepcopy(x_)).reshape(64, 64)

    sub = plt.subplot(2, 2, 2)
    sub.imshow(b_x, cmap='gray')

    # b_x = open_operation(b_x)

    sub = plt.subplot(2, 2, 3)
    sub.imshow(b_x, cmap='gray')
    # plt.show()

    sub = plt.subplot(2, 2, 4)
    sub.imshow(padding(detect(b_x)), cmap='gray')
    plt.show()