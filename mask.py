import cv2
import matplotlib.pyplot as plt
import numpy as np





PATH = r"D:\x.01\images (2).png"
path=PATH
# Load the image
image = cv2.imread(path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding or any other preprocessing steps if required
#ret, threshold = cv2.threshold(gray, 0, 0, 0)

# Find contours in the image
contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over the contours and calculate bounding boxes
x = 0
for i, contour in enumerate(contours):
    # Calculate the bounding box coordinates
    x, y, width, height = cv2.boundingRect(contour)
    blank = np.zeros(gray.shape[:2], dtype = 'uint8')
    # Draw the bounding box rectangle on the image
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    
    foreground = image[y:y+height,x:x+width]

    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

    x_offset=x
    y_offset=y
    blank[y_offset:y_offset+foreground.shape[0], x_offset:x_offset+foreground.shape[1]] = foreground
    
    cv2.imwrite(r'D:\x.01\mask\new{}.jpg'.format(i), blank)

    

# Display the image with the bounding boxes
#cv2.imshow('Image with Contours and Bounding Boxes', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
