import cv2
import matplotlib.pyplot as plt
import numpy as np

##### To-do #####


def fftshift(img):
    '''
    This function should shift the spectrum image to the center.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    row, col = img.shape
    res = img.copy()

    res[0:row//2, 0:col//2] = img[row//2:row, col//2:col]
    res[row//2:row, col//2:col] = img[0:row//2, 0:col//2]
    res[0:row//2, col//2:col] = img[row//2:row, 0:col//2]
    res[row//2:row, 0:col//2] = img[0:row//2, col//2:col]

    return res


def ifftshift(img):
    '''
    This function should do the reverse of what fftshift function does.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    row, col = img.shape
    res = img.copy()

    res[0:row//2, 0:col//2] = img[row//2:row, col//2:col]
    res[row//2:row, col//2:col] = img[0:row//2, 0:col//2]
    res[0:row//2, col//2:col] = img[row//2:row, 0:col//2]
    res[row//2:row, 0:col//2] = img[0:row//2, col//2:col]

    return res


def fm_spectrum(img):
    '''
    This function should get the frequency magnitude spectrum of the input image.
    Make sure that the spectrum image is shifted to the center using the implemented fftshift function.
    You may have to multiply the resultant spectrum by a certain magnitude in order to display it correctly.
    '''
    res = np.log1p(np.abs(fftshift(np.fft.fft2(img))))

    return res


def low_pass_filter(img, r=30):
    '''
    This function should return an image that goes through low-pass filter.
    '''
    row, col = img.shape
    center_row, center_col = row//2, col//2
    fft = fftshift(np.fft.fft2(img))

    for i in range(row):
        for j in range(col):
            if(distance(i, j, center_row, center_col) > r):
                fft[i][j] = 0

    return np.real(np.fft.ifft2(ifftshift(fft)))


def high_pass_filter(img, r=20):
    '''
    This function should return an image that goes through high-pass filter.
    '''

    row, col = img.shape
    center_row, center_col = row//2, col//2
    fft = fftshift(np.fft.fft2(img))

    for i in range(row):
        for j in range(col):
            if(distance(i, j, center_row, center_col) < r):
                fft[i][j] = 0

    return np.real(np.fft.ifft2(ifftshift(fft)))


def denoise1(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''
    row, col = img.shape
    center_row, center_col = row//2, col//2
    fft = fftshift(np.fft.fft2(img))

    point_width = 3
    point_center_1 = 53
    point_center_2 = 81

    point_rows = [center_row+point_center_1, center_row-point_center_1,
                  center_row-point_center_1, center_row+point_center_1]
    point_cols = [center_col+point_center_1, center_col+point_center_1,
                  center_col-point_center_1, center_col-point_center_1]
    for k in range(4):
        for i in range(-point_width, point_width+1):
            for j in range(-point_width, point_width+1):
                fft[point_rows[k]+i][point_cols[k]+j] = 0

    point_rows = [center_row+point_center_2, center_row-point_center_2,
                  center_row-point_center_2, center_row+point_center_2]
    point_cols = [center_col+point_center_2, center_col+point_center_2,
                  center_col-point_center_2, center_col-point_center_2]
    for k in range(4):
        for i in range(-point_width, point_width+1):
            for j in range(-point_width, point_width+1):
                fft[point_rows[k]+i][point_cols[k]+j] = 0

    return np.real(np.fft.ifft2(ifftshift(fft)))


def denoise2(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''
    row, col = img.shape
    center_row, center_col = row//2, col//2
    fft = fftshift(np.fft.fft2(img))

    for i in range(row):
        for j in range(col):
            fft[i][j] = fft[i][j] * \
                butterworth(i, j, center_row, center_col, 28, 2, 10)

    return np.real(np.fft.ifft2(ifftshift(fft)))

#################

# Extra Credit


def dft2(img):
    '''
    Extra Credit. 
    Implement 2D Discrete Fourier Transform.
    Naive implementation runs in O(N^4).
    '''
    row, col = img.shape
    res = np.zeros([row, col], dtype='complex_')
    W_m = np.zeros([row, row], dtype='complex_')
    W_n = np.zeros([col, col], dtype='complex_')

    for u in range(row):
        for x in range(row):
            W_m[u, x] = np.exp(-2j*np.pi*u*x / row)

    for v in range(col):
        for y in range(col):
            W_n[v, y] = np.exp(-2j*np.pi*v*y / col)

    res = np.dot(img, W_n)
    res = np.dot(W_m, res)

    return res


def idft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Discrete Fourier Transform.
    Naive implementation runs in O(N^4). 
    '''
    row, col = img.shape
    res = np.zeros([row, col], dtype='complex_')
    W_x = np.zeros([row, row], dtype='complex_')
    W_y = np.zeros([col, col], dtype='complex_')

    for i in range(row):
        for j in range(row):
            W_x[i, j] = np.exp(2j*np.pi*i*j / row)

    for i in range(col):
        for j in range(col):
            W_y[i, j] = np.exp(2j*np.pi*i*j / col)

    res = np.dot(img, W_y)
    res = np.dot(W_x, res)
    res = res/(row*col)

    return res


def fft2(img):
    '''
    Extra Credit. 
    Implement 2D Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    row, col = img.shape
    res = np.zeros([row, col], dtype='complex_')

    for i in range(row):
        res[i] = calculate_fft(img[i], False)
    res = res.swapaxes(0, 1)
    for i in range(col):
        res[i] = calculate_fft(res[i], False)
    res = res.swapaxes(0, 1)

    return res


def ifft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    row, col = img.shape
    res = np.zeros([row, col], dtype='complex_')

    for i in range(row):
        res[i] = calculate_fft(img[i], True)
    res = res.swapaxes(0, 1)
    for i in range(col):
        res[i] = calculate_fft(res[i], True)
    res = res.swapaxes(0, 1)
    res = res / (row*col)

    return res


def calculate_fft(img, is_inverse):
    N = len(img)
    if(N == 1):
        return img
    else:
        res_even = calculate_fft(img[0::2], is_inverse)
        res_odd = calculate_fft(img[1::2], is_inverse)
        
        kn = []
        for i in range(N):
            kn.append(i)
        kn = np.array(kn)
        
        if(is_inverse):
            W = np.exp(2j*np.pi*kn / N)
        else:
            W = np.exp(-2j*np.pi*kn / N)
        
        res = np.hstack(
            (res_even + W[:int(N/2)]*res_odd,
             res_even + W[int(N/2):]*res_odd))
        
        return res


def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)


def butterworth(x1, y1, x2, y2, D0, w, n):
    D = distance(x1, y1, x2, y2)
    return 1 / (1 + ((D*w) / (D**2 - D0**2))**(2*n))


def gaussian(x1, y1, x2, y2, D0, w):
    D = distance(x1, y1, x2, y2)
    return np.exp(-((D**2-D0**2)/(D*w))**2)


if __name__ == '__main__':
    img = cv2.imread('task2_filtering.png', cv2.IMREAD_GRAYSCALE)
    noised1 = cv2.imread('task2_noised1.png', cv2.IMREAD_GRAYSCALE)
    noised2 = cv2.imread('task2_noised2.png', cv2.IMREAD_GRAYSCALE)

    low_passed = low_pass_filter(img)
    high_passed = high_pass_filter(img)
    denoised1 = denoise1(noised1)
    denoised2 = denoise2(noised2)

    # save the filtered/denoised images
    cv2.imwrite('low_passed.png', low_passed)
    cv2.imwrite('high_passed.png', high_passed)
    cv2.imwrite('denoised1.png', denoised1)
    cv2.imwrite('denoised2.png', denoised2)

    # draw the filtered/denoised images
    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])

    drawFigure((2, 7, 1), img, 'Original')
    drawFigure((2, 7, 2), low_passed, 'Low-pass')
    drawFigure((2, 7, 3), high_passed, 'High-pass')
    drawFigure((2, 7, 4), noised1, 'Noised')
    drawFigure((2, 7, 5), denoised1, 'Denoised')
    drawFigure((2, 7, 6), noised2, 'Noised')
    drawFigure((2, 7, 7), denoised2, 'Denoised')

    drawFigure((2, 7, 8), fm_spectrum(img), 'Spectrum')
    drawFigure((2, 7, 9), fm_spectrum(low_passed), 'Spectrum')
    drawFigure((2, 7, 10), fm_spectrum(high_passed), 'Spectrum')
    drawFigure((2, 7, 11), fm_spectrum(noised1), 'Spectrum')
    drawFigure((2, 7, 12), fm_spectrum(denoised1), 'Spectrum')
    drawFigure((2, 7, 13), fm_spectrum(noised2), 'Spectrum')
    drawFigure((2, 7, 14), fm_spectrum(denoised2), 'Spectrum')

    plt.show()
