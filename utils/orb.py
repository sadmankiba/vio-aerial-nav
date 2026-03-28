import numpy as np
import cv2
from matplotlib import pyplot as plt
 

def find_keypoints_and_descriptors(img_file, grayscale):
    if grayscale:
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_file)
        
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")

    # Initiate ORB detector
    orb = cv2.ORB_create()
    
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    print("Number of keypoints detected: ", len(kp))
    print("Descriptor shape: ", des.shape)
    rand_indices = np.random.choice(len(kp), size=5, replace=False)
    for idx in rand_indices:
        print(f"Keypoint {idx}: Location={kp[idx].pt}, Size={kp[idx].size}, Angle={kp[idx].angle}")
        print(f"Descriptor {idx}: {des[idx]}")

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    if grayscale:
        plt.imshow(img2)
    else:
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    
    plt.savefig(f'orb_keypoints_{"grayscale" if grayscale else "color"}.png')
    
if __name__ == "__main__":
    img_file = '/home/sadman/Git-Repos/ai/ai-dev/sample-data/kitchen-utensils-420x280.png'
    # find_keypoints_and_descriptors(img_file, grayscale=True)
    find_keypoints_and_descriptors(img_file, grayscale=False)
    
    
    
    
"""
Output: 
Image shape: (280, 420, 3), dtype: uint8
Number of keypoints detected:  490
Descriptor shape:  (490, 32)
Keypoint 241: Location=(303.84002685546875, 132.48001098632812), Size=44.6400032043457, Angle=27.066648483276367
Descriptor 241: [  8 179 184  97 234 105 177  95  14  61  60  41 158 191  72  24 183  27
 231  16  18 120 195  80  79 215  53 111 240 200 135 171]
Keypoint 381: Location=(282.0096435546875, 141.00482177734375), Size=64.28160858154297, Angle=120.08363342285156
Descriptor 381: [ 41 253 232  68 253  89 213 255 203 228  60 173  51 172 215  38 137 206
 213  45 204  43 180 153 127  93   5  63 199 252 214 254]
Keypoint 271: Location=(295.20001220703125, 139.6800079345703), Size=44.6400032043457, Angle=267.5135192871094
Descriptor 271: [ 54 170  18 153 195 177 160 194  46  19 224 227   0 138 100 136 233 155
 130 145  22 191   6 241  20 196 154  67 209   1 171 134]
Keypoint 423: Location=(333.4349670410156, 144.32260131835938), Size=77.137939453125, Angle=320.32586669921875
Descriptor 423: [ 28  76 228 235 159 127  70 156 199 146 142   2 251 125  57 230 121 210
 255 250 106 165 123 191 235 207 133  21 243 255 116  34]
Keypoint 336: Location=(127.87200927734375, 153.79200744628906), Size=53.5680046081543, Angle=108.40039825439453
Descriptor 336: [ 17 135  77 137 150  34  74  63 100  32 202 230  83  34 152 229 114 134
  12 172 185 152  69  13 184 168 162 192  61 202 240  44]
"""