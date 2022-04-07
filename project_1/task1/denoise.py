import cv2
import numpy as np
from sympy import false
from tqdm import tqdm


def task1_2(src_path, clean_path, dst_path):
    """
    This is main function for task 1.
    It takes 3 arguments,
    'src_path' is path for source image.
    'clean_path' is path for clean image.
    'dst_path' is path for output image, where your result image should be saved.

    You should load image in 'src_path', and then perform task 1-2,
    and then save your result image to 'dst_path'.
    """
    noisy_img = cv2.imread(src_path)
    clean_img = cv2.imread(clean_path)

    # do noise removal
    sigma_s = 200
    sigma_r = 70
    # result_img = apply_median_filter(noisy_img, kernel_size=3)
    # result_img = apply_bilateral_filter(
    #     noisy_img, 5, sigma_s, sigma_r)
    # result_img = apply_my_filter(noisy_img)

    # print("rms: ", calculate_rms(clean_img, result_img))
    # cv2.imwrite(dst_path, result_img)

    # find optimal solution
    kernel_size = 3
    result_img_med = apply_median_filter(noisy_img, kernel_size)
    rms_med = calculate_rms(clean_img, result_img_med)
    result_img_bi = apply_bilateral_filter(noisy_img, kernel_size, 200, 70)
    rms_bi = calculate_rms(clean_img, result_img_bi)
    result_img_my = apply_my_filter(noisy_img)
    rms_my = calculate_rms(clean_img, result_img_my)

    min_rms = min(rms_med, rms_bi, rms_my)
    if(min_rms == rms_med):
        while(1):
            res = apply_median_filter(noisy_img, kernel_size+2)
            rms = calculate_rms(clean_img, res)
            if(rms > rms_med):
                break
            rms_med = rms
            result_img_med = res
            kernel_size += 2
        print("rms: ", rms_med, " / type: median / kernel size: ", kernel_size)
        cv2.imwrite(dst_path, result_img_med)
    elif(min_rms == rms_bi):
        while(1):
            res = apply_bilateral_filter(noisy_img, kernel_size+2, 200, 70)
            rms = calculate_rms(clean_img, res)
            if(rms > rms_bi):
                break
            rms_bi = rms
            result_img_bi = res
            kernel_size += 2
        print("rms: ", rms_bi, " / type: bilateral / kernel size: ", kernel_size)
        cv2.imwrite(dst_path, result_img_bi)
    else:
        print("rms: ", rms_my, " / type: my / kernel size: ", 5)
        cv2.imwrite(dst_path, result_img_my)
    pass


def apply_median_filter(img, kernel_size):
    """
    You should implement median filter using convolution in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is an int value, which determines kernel size of median filter.

    You should return result image.
    """
    row, col, channel = img.shape

    input = img.copy()
    res = img.copy()
    pad_num = kernel_size//2

    input = img_padding(input, row, pad_num)

    for i in tqdm(range(row)):
        for j in range(col):
            for k in range(channel):
                res[i][j][k] = np.median(
                    input[i:i+kernel_size, j:j+kernel_size, k])

                # if(j+kernel_size>=col+kernel_size-1):
                #     avg = np.mean(input[i:i+kernel_size, j:-1, k])
                # else:
                #     avg = np.mean(input[i:i+kernel_size, j:j+kernel_size+1, k])

                # is_value_bigger = True
                # for x in range(kernel_size):
                #     for y in range(kernel_size+1):
                #         coord_col = j+y
                #         if(coord_col>=col+kernel_size-1):
                #             coord_col = -1
                #         if(x!=pad_num and y!=pad_num and input[i+x][coord_col][k]<avg):
                #             is_value_bigger = False
                #             break
                # if(is_value_bigger):
                #     input[i+pad_num][j+pad_num][k] = np.median(input[i:i+kernel_size, j:j+kernel_size, k])
                #     res[i][j][k] = input[i+pad_num][j+pad_num][k]

    return res


def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    """
    You should implement bilateral filter using convolution in this function.
    It takes at least 4 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of average filter.
    'sigma_s' is a int value, which is a sigma value for G_s(gaussian function for space)
    'sigma_r' is a int value, which is a sigma value for G_r(gaussian function for range)

    You should return result image.
    """
    row, col, channel = img.shape

    input = img.copy()
    res = img.copy()
    pad_num = kernel_size//2

    input = img_padding(input, row, pad_num)

    for i in tqdm(range(row)):
        for j in range(col):
            for k in range(channel):
                calulate_Ip(input, res, kernel_size, i, j, k, sigma_s, sigma_r)

    return res


def apply_my_filter(img):
    """
    You should implement additional filter using convolution.
    You can use any filters for this function, except median, bilateral filter.
    You can add more arguments for this function if you need.

    You should return result image.
    """
    row, col, channel = img.shape

    input = img.copy()
    res = img.copy()
    pad_num = 5//2

    input = img_padding(input, row, pad_num)

    # Nagao-Matsuyama filter
    for i in tqdm(range(row)):
        for j in range(col):
            for k in range(channel):
                calculate_NM(input, res, i, j, k)

    return res


def calculate_rms(img1, img2):
    """
    Calculates RMS error between two images. Two images should have same sizes.
    """
    if (img1.shape[0] != img2.shape[0]) or \
            (img1.shape[1] != img2.shape[1]) or \
            (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have same sizes.")

    diff = np.abs(img1 - img2)
    diff = np.abs(img1.astype(dtype=np.int) - img2.astype(dtype=np.int))
    return np.sqrt(np.mean(diff ** 2))


def img_padding(img, row, pad_num):
    input = img.tolist()

    for i in range(row):
        col_start = img[i][0]
        col_end = img[i][-1]
        for j in range(pad_num):
            input[i].insert(0, col_start)
            input[i].append(col_end)
    row_start = input[0]
    row_end = input[-1]
    for j in range(pad_num):
        input.insert(0, row_start)
        input.append(row_end)

    return np.array(input)


def G(x, sigma):
    return (1.0 / np.sqrt(2 * np.pi * (sigma ** 2))) * np.exp(- (x ** 2) / (2 * sigma ** 2))


def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def calulate_Ip(input, img, kernel_size, coord_row, coord_col, channel, sigma_s, sigma_r):
    pad_num = kernel_size//2
    p_row = coord_row+pad_num
    p_col = coord_col+pad_num
    Wp = 0
    res = 0

    for i in range(kernel_size):
        for j in range(kernel_size):
            q_row = coord_row+i
            q_col = coord_col+j
            Gs = G(distance(p_row, p_col, q_row, q_col), sigma_s)
            Gr = G(np.abs(input[p_row][p_col][channel] -
                   input[q_row][q_col][channel]), sigma_r)
            w = Gs*Gr
            Wp += w
            res += w*input[q_row][q_col][channel]

    res = res/Wp
    img[coord_row][coord_col][channel] = int(round(res))


def calculate_NM(input, img, coord_row, coord_col, channel):
    vars = np.array([])
    areas = []

    area1 = input[coord_row:coord_row+2,
                  coord_col+1:coord_col+4, channel].flatten()
    area1 = np.append(area1, input[coord_row+2, coord_col+2, channel])
    areas.append(area1)
    vars = np.append(vars, np.var(area1))

    area2 = input[coord_row+1:coord_row+4,
                  coord_col+3:coord_col+5, channel].flatten()
    area2 = np.append(area2, input[coord_row+2, coord_col+2])
    areas.append(area2)
    vars = np.append(vars, np.var(area2))

    area3 = input[coord_row+3:coord_row+5,
                  coord_col+1:coord_col+4, channel].flatten()
    area3 = np.append(area3, input[coord_row+2, coord_col+2])
    areas.append(area3)
    vars = np.append(vars, np.var(area3))

    area4 = input[coord_row+1:coord_row+4,
                  coord_col:coord_col+2, channel].flatten()
    area4 = np.append(area4, input[coord_row+2, coord_col+2])
    areas.append(area4)
    vars = np.append(vars, np.var(area4))

    area5 = input[coord_row:coord_row+2,
                  coord_col:coord_col+2, channel].flatten()
    area5 = np.append(area5, input[coord_row+2, coord_col+2])
    areas.append(area5)
    vars = np.append(vars, np.var(area5))

    area6 = input[coord_row:coord_row+2,
                  coord_col+3:coord_col+5, channel].flatten()
    area6 = np.append(area6, input[coord_row+2, coord_col+2])
    areas.append(area6)
    vars = np.append(vars, np.var(area6))

    area7 = input[coord_row+3:coord_row+5, coord_col+3:coord_col+5].flatten()
    area7 = np.append(area7, input[coord_row+2, coord_col+2])
    areas.append(area7)
    vars = np.append(vars, np.var(area7))

    area8 = input[coord_row+3:coord_row+5, coord_col:coord_col+2].flatten()
    area8 = np.append(area8, input[coord_row+2, coord_col+2])
    areas.append(area8)
    vars = np.append(vars, np.var(area8))

    min_var = np.min(vars)
    for i in range(len(vars)):
        if(vars[i] == min_var):
            img[coord_row][coord_col][channel] = int(round(np.mean(areas[i])))
            break


if __name__ == '__main__':
    task1_2('test_images/cat_noisy.jpg',
            'test_images/cat_clean.jpg', 'res/cat_clean.jpg')
    # task1_2('test_images/fox_noisy.jpg',
    #         'test_images/fox_clean.jpg', 'res/fox_clean.jpg')
    # task1_2('test_images/Snowman_noisy.jpg',
    #         'test_images/Snowman_clean.jpg', 'res/Snowman_clean.jpg')
