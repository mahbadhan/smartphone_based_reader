from PIL import Image

im = Image.open('strip_photo.JPG')

width, height = im.size
print(im.size)
# im.show()

x = width / 2
y = height / 2

if width > height:
    cropp = im.crop((x - 200, y - 200, x, y + 350))
    cropp2 = im.crop((x, y - 200, x + 200, y + 350))
    # cropp.show()
    # cropp2.show()
else:
    cropp = im.crop((x -200, y-200 , x + 200, y ))
    cropp2 = im.crop((x -200, y , x + 200, y + 200 ))
    cropp.show()
    cropp2.show()



cropp.save("b1_crop.jpg")
cropp2.save("b2_crop.jpg")


import cv2

import numpy as np

image1 = cv2.imread('b1_crop.jpg')
image2 = cv2.imread('b2_crop.jpg')


def segment(image):
    result = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([170, 55, 25 ])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower, upper)
    result = cv2.bitwise_and(result, result, mask=mask)
    cv2.imshow('result', result)
    cv2.waitKey()

    return result


masked1 = segment(image1)
cv2.imwrite('C:/Users/Mahima Badhan/PycharmProjects/untitled/mask1.jpg', masked1)

masked2 = segment(image2)
cv2.imwrite('C:/Users/Mahima Badhan/PycharmProjects/untitled/mask2.jpg', masked2)

from PIL import Image

im11 = Image.open('mask1.jpg')

im22 = Image.open('mask2.jpg')


def intensity(imm):
    width, height = imm.size
    orig_pixel_map = imm.load()

    sum = 0

    for w in range(width):
        for h in range(height):
            x, y, z = orig_pixel_map[w, h]
            sum = sum + x

    return sum


T = intensity(im11)
print("summation of red pixels intensity at T line is", T)
C = intensity(im22)
print("summation of red pixels intensity  at C line is", C)

ratio1 = T / C
print("ratio of test to control line signal intensity is", ratio1)


import matplotlib.pyplot as plt
import pandas as pd

import numpy as np




df = pd.read_csv("testfile.csv")

df.head()
cdf = df[['ratio', 'result']]
cdf.head(9)

viz = cdf[['ratio', 'result']]
viz.hist()
#plt.show()

plt.scatter(cdf.ratio, cdf.result,  color='blue')
plt.xlabel("ratio")
plt.ylabel("result")
#plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.ratio, train.result,  color='blue')
plt.xlabel("ratio")
plt.ylabel("result")
#plt.show()

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ratio']])
train_y = np.asanyarray(train[['result']])
regr.fit(train_x, train_y)

print('Coefficients: ', regr.coef_)
print('Intercept: ',regr.intercept_)

plt.scatter(train.ratio, train.result,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("ratio")
plt.ylabel("result")
#plt.show()

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ratio']])
test_y = np.asanyarray(test[['result']])
test_y_hat = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


z = regr.coef_[0][0] * ratio1 + regr.intercept_[0]
print("the concentration of analyte in the sample is", z)