import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
import cv2

# RLE ENCODER
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RLE DECODER
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

'''
GET TRAIN CSV

'''
train_df=pd.read_csv('./train.csv')

img_id=[]
img_path=[]
mask_rle=[]

# img_id,img_path,mask_rle
df=pd.DataFrame({'img_id':img_id,'img_path':img_path,'mask_rle':mask_rle})
df.to_csv('output.csv',index=False)

'''
SLICE MASK
'''
for img_cnt in range(7140): # 7140
    mask = rle_decode(train_df.iloc[img_cnt,2], (1024, 1024))
    cnt=0
    
    print(img_cnt)

    for i in range(4):
        for j in range(4):
            sliced_mask=mask[i*256 : (i+1)*256,
                             j*256 : (j+1)*256]
        
            img_id.append('TRAIN_{0:04d}'.format(img_cnt*16+cnt))
            img_path.append('./train_img/TRAIN_{0:04d}.png'.format(img_cnt*16+cnt))
            mask_rle.append(rle_encode(sliced_mask))

            cnt+=1

# img_id,img_path,mask_rle
df=pd.DataFrame({'img_id':img_id,'img_path':img_path,'mask_rle':mask_rle})
df.to_csv('train_edit.csv',index=False)