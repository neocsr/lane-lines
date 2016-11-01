import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

print(cv2.__version__)

image = mpimg.imread('test_images/solidWhiteRight.jpg')
print('Image is:', type(image), 'with dimensions:', image.shape)

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=1):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for i, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold,
        np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def slope(x1, y1, x2, y2):
    """
    Calculated the slope between two points
    """
    m = (y2 - y1)/(x2 - x1)
    return m

def draw_lines_enhanced(img, lines, color=[255, 0, 0], thickness=10):
    """
    It assumes two different lines `y = mx + b`, one with negative slope
    `left_m` and the other positive `right_m`.
    Each line has its own intercept`left_b` and `right_b`.

    The slope and intercept are generated with the `fitLine` function.
    """
    left_points = []
    right_points = []
    y_min = img.shape[0]
    y_max = img.shape[0]

    for line in lines:
        x1, y1, x2, y2 = line[0]
        m = slope(x1, y1, x2, y2)
        if m < 0:
            left_points.append((x1, y1))
            left_points.append((x2, y2))
        else:
            right_points.append((x1, y1))
            right_points.append((x2, y2))

        if (y1 < y2):
            y_min = y_min if (y_min < y1) else y1
        else:
            y_min = y_min if (y_min < y2) else y2

    vx, vy, x0, y0 = cv2.fitLine(
        np.array(left_points), cv2.DIST_L2, 0, 0.01, 0.01)
    left_m = vy/vx
    left_b = y0 - left_m * x0
    left_x_min = int((y_min - left_b)/left_m)
    left_x_max = int((y_max - left_b)/left_m)
    cv2.line(img,
        (left_x_min, y_min),
        (left_x_max, y_max), [155, 50, 0], thickness)

    vx, vy, x0, y0 = cv2.fitLine(
        np.array(right_points), cv2.DIST_L2, 0, 0.01, 0.01)
    right_m = vy/vx
    right_b = y0 - right_m * x0
    right_x_min = int((y_min - right_b)/right_m)
    right_x_max = int((y_max - right_b)/right_m)
    cv2.line(img,
        (right_x_min, y_min),
        (right_x_max, y_max), color, thickness)

def hough_lines_enhanced(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold,
        np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines_enhanced(line_img, lines)
    return line_img

def pipeline(image):
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, 5)
    edges = canny(blur_gray, 50, 150)

    y_size = edges.shape[0]
    x_size = edges.shape[1]
    vertices = np.array([[
        (0, y_size),
        (x_size/2 - 20, y_size/2 + 50),
        (x_size/2 + 20, y_size/2 + 50),
        (x_size, y_size)]], dtype=np.int32)
    mask = region_of_interest(edges, vertices)

    hough = hough_lines(
        mask, rho=1, theta=1,
        threshold=10, min_line_len=20, max_line_gap=40)

    hough_enhanced = hough_lines_enhanced(
        mask, rho=1, theta=1,
        threshold=10, min_line_len=20, max_line_gap=40)

    result = weighted_img(hough_enhanced, image)

    plt.figure(figsize=(18, 12))
    plt.subplot(2, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.title("Canny with region mask")
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(hough, cmap='gray')
    plt.title("Hough")
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(hough_enhanced, cmap='gray')
    plt.title("Hough with linear fit")
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(result, cmap='gray')
    plt.title("Original with overlay")
    plt.axis('off')
    plt.show()

    return result

pipeline(image)
