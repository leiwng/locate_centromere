import cv2
import os
import numpy as np
from scipy.interpolate import interp1d

def smooth_contour(contour):
    if contour.shape[0] > 20:
        # 拟合轮廓
        contour = contour.squeeze()
        # 对x轴坐标和y轴坐标分别进行一维插值
        f_x = interp1d(range(len(contour)), contour[:, 0], kind='cubic')
        f_y = interp1d(range(len(contour)), contour[:, 1], kind='cubic')

        # smooth_factor平滑因子 mask_高斯核
        smooth_factor = 2
        mask_ = 7
        num_points = len(contour)
        t = np.linspace(0, num_points - 1, int(num_points / smooth_factor))
        x_smooth = cv2.GaussianBlur(f_x(t), (mask_, mask_), 0)
        y_smooth = cv2.GaussianBlur(f_y(t), (mask_, mask_), 0)
        smoothed_points = np.stack((x_smooth, y_smooth), axis=1).astype(np.int32).reshape((-1, 1, 2))

        smoothcontour = [smoothed_points]
        return smoothcontour

def smooth_image(img):
    '''

    :param img: RGB image
    :return: smoothed RGB image
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    c, h = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # only the maximum contour is needed
    countour_i = 0
    for i in range(1, len(c)):
        if cv2.contourArea(c[i]) > cv2.contourArea(c[countour_i]):
            countour_i = i
    c = c[countour_i]
    c = smooth_contour(c)
    mask = np.full(shape=(img.shape[0], img.shape[1], 3),
                   fill_value=1,
                   dtype='uint8')
    cv2.drawContours(mask, c, -1, (255, 255, 255), cv2.FILLED)
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

    res = cv2.bitwise_and(img, img, mask=mask)
    res = cv2.add(res, img)
    return res


