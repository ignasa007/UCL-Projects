import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axs = plt.subplots(1, 1, figsize=(8, 7))

img = mpimg.imread('figure-1.jpg')
height, width, channels = img.shape
del_h, del_w, scale, diff = 49, 61, 26.5, 3
axs.imshow(img, extent=[
    (-width/2-del_w)/scale-0.2,
    (width/2-del_w)/scale,
    (-height/2-del_h)/(scale+diff)+0.2,
    (height/2-del_h)/(scale+diff), 
])

circle = plt.Circle((0, 0), 5, color='green', fill=False, linewidth=2, linestyle='--')
axs.add_patch(circle)

plt.axis('off')
fig.tight_layout()
plt.savefig('assets/q1-boundary.png')
plt.close(fig)