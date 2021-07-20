from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image

# Set image names
im1 = mpimg.imread("airplane_data/planes.jpg")
target1 = mpimg.imread("target-0.png")
mask1 = mpimg.imread("mask-0.png")

# Create figure
plt.imshow(im1)
plt.title('Image')
plt.savefig('images.png')
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(target1)
ax.set_title('Ground Truth')
ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(mask1)
ax.set_title('Predicted Mask')
plt.savefig('masks.png')
