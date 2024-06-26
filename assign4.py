import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# Q 1

# Loading data from book1.csv
array = np.genfromtxt('book1.csv', skip_header=1, delimiter='\t', usecols=(1))
print(array)

# Calculating the maximum and minimum values from the array
max_value = np.max(array)
min_value = np.min(array)
print(f"max = {max_value}")
print(f"max = {min_value}")

# Q 2

# Sorting the array
array_sort = np.sort(array)
print(array_sort)


# Q 3

# Reversing the array
array_rev = array[::-1]
print(array_rev)

# Q 4

# Loading data from multiple files, calculating mean for each array and printing the means
files=["book1.csv","book2.csv","book3.csv"]
arrlist=[np.genfromtxt(file, skip_header=1, delimiter='\t',usecols=(1)) for file in files ]
means = [np.mean(array) for array in arrlist]
print(f"Means of Arrays: {means}")

# Q 5
img=cv2.imread('a.png', cv2.IMREAD_COLOR)
X=np.array(img)
cv2.imshow("output",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Q 6
grayimg=cv2.cvtColor(X,cv2.COLOR_BGR2GRAY)
arrayX = np.array(grayimg)
cv2.imshow("output",grayimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Q 7
arrayY = arrayX.T
arrayZ = np.matmul(arrayX,arrayY)
print(arrayZ)

# Q 8
start_time = time.time()
arrayZ = np.matmul(arrayX,arrayY)
end_time = time.time()
print(f"Time taken with NumPy: {end_time - start_time} seconds")

start_time = time.time()

res=np.dot(arrayX,arrayY)
end_time = time.time()
print(f"Time taken without NumPy: {end_time - start_time} seconds")

# Q 9

# Plotting the histogram of pixel intensities in the grayscale image
plt.hist(grayimg.ravel(), bins=256, range=[0, 256], color='gray')

# Adding title and labels to the histogram plot
plt.title('Pixel Intensity Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Displaying the histogram
plt.show()


# Q 10

# Drawing a filled rectangle on the grayscale image
rimg = cv2.rectangle(grayimg, (40, 100), (70, 200), (0, 0, 0), -1)

# Displaying the image with the drawn rectangle
cv2.imshow("Image with Rectangle", rimg)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Q 11

# List of threshold values to be applied to the grayscale image
thresholds = [50, 70, 100, 150]

# Binarizing the image using different threshold values and storing results in a list

binarized_images = [cv2.threshold(grayimg, thresh, 255, cv2.THRESH_BINARY)[1] for thresh in thresholds]

# Displaying the binarized images with corresponding threshold values
for i, thresh in enumerate(thresholds):
    cv2.imshow(f'Z {thresh}', binarized_images[i])
    cv2.waitKey(0)
    
# Closing all open windows
cv2.destroyAllWindows()


# Q 12

# Define a simple 3x3 filter for image convolution
filter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Applying the filter to the input image using cv2.filter2D
filtered_img = cv2.filter2D(src=img, ddepth=-1, kernel=filter)

# Displaying the filtered images
cv2.imshow('Filtered Image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
