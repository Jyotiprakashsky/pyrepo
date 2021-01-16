import cv2
import pandas as pd
import numpy as np
from numpy.linalg import norm
import glob
import pickle
# local modules
#RandomForestClassifier
def RandomForest_learn(x,y,X_test,Y_test):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=3, criterion='entropy', random_state=0)
    classifier.fit(x, y)
    y_pred = classifier.predict(X_test)
    pickle.dump(classifier, open('RandomForestClassifier.sav', 'wb'))
    n = 0
    p = 0
    i = 0
    while i < len(y_pred):
        if y_pred[i] == Y_test[i]:
            p = p + 1
        else:
            n = n + 1
        i = i + 1
    print(n)
    print(p)
    accuracy = (p / len(y_pred)) * 100
    print(accuracy, '% RandomForestClassifier')


#DecisionTreeClassifier
def DecisionTree_learn(x,y,X_test,Y_test):
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(x, y)
    y_pred = classifier.predict(X_test)
    pickle.dump(classifier, open('DecisionTreeClassifier.sav', 'wb'))
    n = 0
    p = 0
    i = 0
    while i < len(y_pred):
        if y_pred[i] == Y_test[i]:
            p = p + 1
        else:
            n = n + 1
        i = i + 1
    print(n)
    print(p)
    accuracy = (p / len(y_pred)) * 100
    print(accuracy, '% DecisionTreeClassifier')




#Kernel SVM classification
def KernelSVM_learn(x,y,X_test,Y_test):
    # Fitting SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf', gamma=5.383, C=2.67)
    classifier.fit(x, y)
    y_pred = classifier.predict(X_test)
    pickle.dump(classifier, open('KernelSVMClassifier.sav', 'wb'))
    n = 0
    p = 0
    i = 0
    while i < len(y_pred):
        if y_pred[i] == Y_test[i]:
            p = p + 1
        else:
            n = n + 1
        i = i + 1
    print(n)
    print(p)
    accuracy = (p / len(y_pred)) * 100
    print(accuracy, '% KernelSVM')



#Linear SVM classification
def LinearSVM_learn(x,y,X_test,Y_test):
    # Fitting SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(x, y)
    y_pred = classifier.predict(X_test)
    pickle.dump(classifier, open('LinearSVMClassifier.sav', 'wb'))
    n = 0
    p = 0
    i = 0
    while i < len(y_pred):
        if y_pred[i] == Y_test[i]:
            p = p + 1
        else:
            n = n + 1
        i = i + 1
    print(n)
    print(p)
    accuracy = (p / len(y_pred)) * 100
    print(accuracy, '% linearSVM')


#logit classification
def Logit_learn(x,y,X_test,Y_test):
    # Fitting Logit to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(x, y)
    y_pred = classifier.predict(X_test)
    pickle.dump(classifier, open('LogisticClassifier.sav', 'wb'))
    n = 0
    p = 0
    i = 0
    while i < len(y_pred):
        if y_pred[i] == Y_test[i]:
            p = p + 1
        else:
            n = n + 1
        i = i + 1
    print(n)
    print(p)
    accuracy = (p / len(y_pred)) * 100
    print(accuracy, '% logit')


#KNN classification
def KNN_learn(x,y,X_test,Y_test):
    # Fitting K-NN to the Training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(x, y)
    y_pred = classifier.predict(X_test)
    pickle.dump(classifier, open('KNeighborsClassifier.sav', 'wb'))
    n = 0
    p = 0
    i = 0
    while i < len(y_pred):
        if y_pred[i] == Y_test[i]:
            p = p + 1
        else:
            n = n + 1
        i = i + 1
    print(n)
    print(p)
    accuracy = (p / len(y_pred)) * 100
    print(accuracy, '%')

def lableName(valueL, indexV):
	val = []
	if valueL == 1:
		for i in range(indexV):
			val.append(1)
		return val
	else:
		for i in range(indexV):
			val.append(0)
		return val

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        print(hist)

        samples.append(hist)
    return np.float32(samples)

test_images_ships = glob.glob('D:\\IMGprocesingOpenCV\\HOG DIgits\\dataIMG\\two\*.png')
i=0;
Imges = []
for iname in test_images_ships:
    i = i + 1
    image = cv2.imread(iname)
    image = cv2.resize(image, (64, 64))
    Imges.append(image)


samples = preprocess_hog(Imges)
print('#############################################################################################')
print(samples)


test_images_non_ships = glob.glob('D:\\IMGprocesingOpenCV\\HOG DIgits\\dataIMG\\gr\*.png')
i=0;
Imges = []
for iname in test_images_non_ships:
    i = i + 1
    image = cv2.imread(iname)
    image = cv2.resize(image, (64, 64))
    Imges.append(image)


Rsamples = preprocess_hog(Imges)
print('#############################################################################################')
print(Rsamples)
df1 = pd.DataFrame(Rsamples)
df1['lable'] = lableName(1,df1.shape[0])
df2 = pd.DataFrame(samples)
df2['lable'] = lableName(0,df2.shape[0])
frames = [df1, df2]
df = pd.concat(frames)
from sklearn.utils import shuffle
df = shuffle(df)
y = df.iloc[:800, 64].values
x = df.iloc[:800, :64].values
X_test = df.iloc[800:, :64].values
Y_test = df.iloc[800:, 64].values
KNN_learn(x,y,X_test,Y_test)
Logit_learn(x,y,X_test,Y_test)
LinearSVM_learn(x,y,X_test,Y_test)
KernelSVM_learn(x,y,X_test,Y_test)
DecisionTree_learn(x,y,X_test,Y_test)
RandomForest_learn(x,y,X_test,Y_test)
print(df.shape[0],'...............')
df.to_csv('hog.csv')
