import numpy as np
from PIL import Image

im = Image.open("/home/jerinpaul/Pictures/mountains.jpg")
im_arr = np.array(im)
im.show()
print(im_arr)
