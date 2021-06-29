from PIL import Image

im = Image.open('C:/Users/Mahima Badhan/PycharmProjects/untitled/30.jpg')

#size and image of the LFA strip
width, height = im.size
print(im.size)
#im.show()

x = width / 2
y = height / 2


#cropping the image around the T and C line
cropp = im.crop((x - 200, y - 200, x, y + 350))
cropp2 = im.crop((x, y - 200, x + 200, y + 350))
#cropp.show()
#cropp2.show()



#two separate images of T line and C line
cropp.save("b1_crop.jpg")
cropp2.save("b2_crop.jpg")


#segmenting the cropped lines such that only the coloured lines are masked out
import cv2

import numpy as np

image1 = cv2.imread('b1_crop.jpg')
image2 = cv2.imread('b2_crop.jpg')


def segment(image):
    result = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([137, 42, 70])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower, upper)
    result = cv2.bitwise_and(result, result, mask=mask)
    cv2.imshow('masked image of T and C line', result)
    cv2.waitKey()

    return result


masked1 = segment(image1)
cv2.imwrite('C:/Users/Mahima Badhan/PycharmProjects/untitled/mask1.jpg', masked1)

masked2 = segment(image2)
cv2.imwrite('C:/Users/Mahima Badhan/PycharmProjects/untitled/mask2.jpg', masked2)

#Calculating the sum of red pixels intesity of the lines
#the suum of red pixels intensity is directly proportional to the concentration of analyte.
from PIL import Image

im11 = Image.open('mask1.jpg')

im22 = Image.open('mask2.jpg')


def intensity(imm):
    w, h = imm.size
    pix = imm.load()

    r_sum = 0
    b_sum = 0
    g_sum = 0
    n = 0

    for i in range(0,w):
        for j in range(0,h):
            p = pix[i,j]
            r_sum += p[0]
            g_sum += p[1]
            b_sum += p[2]
            if p[0]!= 0:
                n += 1


    sum = r_sum + g_sum + b_sum
    sum =sum/n

    return sum


C = intensity(im11)
#print("summation of red pixels intensity at C line is", C)
T = intensity(im22)
#print("summation of red pixels intensity  at T line is", T)


#calculating the ratio of red pixels intensity of T line to C line
#Ratio is also proportional to the conc. of target analyte
#the ratio gives a normalised result even if the image is clicked in different illuminating conditions.
ratio1 = T / C
#print("ratio of test to control line signal intensity is", ratio1)

#Calibration using Machine Learning
import matplotlib.pyplot as plt
import pandas as pd



#the known data  file imported
#File containing data regarding the concentration of analytes of known samples and their T/C intensity ratio
df = pd.read_csv("total_trial.csv")

df.head()
cdf = df[['ratio', 'result']]
cdf.head(9)


#plotting
plt.scatter(cdf.ratio, cdf.result,  color='blue')
plt.xlabel("ratio")
plt.ylabel("concenration")
#plt.show()

#training data and test data classified from the file
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


plt.scatter(train.ratio, train.result,  color='blue')
plt.xlabel("ratio")
plt.ylabel("concentration")
#plt.show()


#Machine learning algorithm to apply linear regression model on the train data.
#coefficient and intercept obtained from the regression analysis
#plotting the graph
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ratio']])
train_y = np.asanyarray(train[['result']])
regr.fit(train_x, train_y)

print('Coefficients of the ratio-conc plot ', regr.coef_)
print('Intercept of ratio-conc plot ',regr.intercept_)

plt.scatter(train.ratio, train.result,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("ratio of intensity at T to C line")
plt.ylabel("concentration of analyte")
#plt.show()

#Machine learning algorithm to predict the concentration of analyte for the test data
#Calculating the error among the predicted conc. and the actual conc. of the test data
from sklearn.metrics import r2_score


test_x = np.asanyarray(test[['ratio']])
test_y = np.asanyarray(test[['result']])
test_y_hat = regr.predict(test_x)
#print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
#print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
#print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )

#For every T/C intensity ratios, the concentration of the analyte is predicted using the parameters of the linear regression analysis
z = regr.coef_[0][0] * ratio1 + regr.intercept_[0]
print("the concentration of analyte in the sample is", z)
