from pyk4a import Config, PyK4A
from matplotlib import pyplot as plt
import cv2

# Load camera with the default config
k4a = PyK4A()
k4a.start()

# Get the next capture (blocking function)
capture = k4a.get_capture()
img_color = capture.color
cv2.imwrite("ooo.jpg", img_color)

# Display with pyplot
plt.imshow(img_color[:, :, 2::-1]) # BGRA to RGB
plt.show()
