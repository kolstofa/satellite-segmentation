import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import cv2

root_dir='./train_img/'
for img_cnt in range(7140): #7140
    large_image='TRAIN_{0:04d}.png'.format(img_cnt)
    print(large_image)
    img=cv2.imread(root_dir+large_image)

    patches_img=patchify(img,(256,256,3),step=256)

    extra_cnt=0
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img=patches_img[i,j,0,:,:,:]
            if not cv2.imwrite('sliced_img/TRAIN_{0:04d}.png'.format(img_cnt*16+extra_cnt),single_patch_img):
                raise Exception("Could not write the image")
            extra_cnt+=1