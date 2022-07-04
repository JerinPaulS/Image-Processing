import numpy as np
import nibabel as nib
import itk
import itkwidgets
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

image_path = "/media/jerinpaul/New Volume/Datasets/Brain MRI/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz"
image_obj = nib.load(image_path)
image_data = image_obj.get_fdata()

label_path = "/media/jerinpaul/New Volume/Datasets/Brain MRI/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz"
label_obj = nib.load(label_path)
label_data = label_obj.get_fdata()

print("Categories are: ", (np.unique(label_data)))

layer = 50

classes_dict = {
    'Normal': 0.,
    'Edema': 1.,
    'Non-enhancing tumor': 2.,
    'Enhancing tumor': 3.
}

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(40, 25))
for i in range(4):
    img_label_str = list(classes_dict.keys())[i]
    img = label_data[:, :, layer]
    mask = np.where(img == classes_dict[img_label_str], 255, 0)
    ax[i].imshow(mask)
    ax[i].set_title(f'Layer {layer} for {img_label_str}')
    ax[i].axis('off')
plt.tight_layout()
plt.show()

'''
height, width, depth, channels = image_data.shape
print("The dimensions of the image are: ", (height, width, depth, channels))

maxval = 154
i = np.random.randint(0, maxval)
channel = 0
print(f'Plotting layer Layer {i}, Channel {channel} of image.')
plt.imshow(image_data[:, :, i, channel], cmap='gray')
plt.axis('off')
plt.show()

def view3Dimage(layer):
    plt.figure(figsize=(10, 15))
    channel = 3
    plt.imshow(image_data[:, :, i, channel], cmap='gray')
    plt.title('View layers of Brain MRI')
    plt.axis('off')
    return layer

interact(view3Dimage, layer=(0, image_data.shape[2] - 1))
'''