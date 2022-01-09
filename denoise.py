import argparse
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser(description='Denoise Process')
parser.add_argument('-p', type=str, help='input image path')

args = parser.parse_args()

if args.p is None:
    print('Please input image path by -p xxxx.png')
    exit()

image = cv2.imread(args.p, 0)

denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
img_norm = cv2.normalize(denoised, dst=None, alpha=350, beta=10, norm_type=cv2.NORM_MINMAX)
img_norm = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

cv2.imwrite(os.path.splitext(args.p)[0] + '_denoised.png', img_norm)
