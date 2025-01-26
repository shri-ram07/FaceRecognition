from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split , LeaveOneOut , cross_val_score
import numpy as np
from sklearn.decomposition import PCA

faces = datasets.fetch_olivetti_faces()
features = faces.data
targets = faces.target



#if you what to visualise the faces then run the code written below
"""#Now we will visualise all the 10 persons image through pyplot
fig , sub_plot = plt.subplots(nrows = 5, ncols = 8, figsize = (20,20))   # it will generate 40 subplots
sub_plot = sub_plot.flatten()   # to plot all subplots in a single plot or convert it to one dimensional array

for i in np.unique(targets):
    face_index = i*10
    sub_plot[i].imshow(features[face_index].reshape(64,64),cmap="gray")
    sub_plot[i].set_xticks([])
    sub_plot[i].set_yticks([])
    sub_plot[i].set_title("Face Id is : %s"%i )
plt.suptitle("The Dataset if 40 faces")
plt.show()"""




# split the datasets
train_x, test_x, train_y, test_y = train_test_split(features, targets, test_size = 0.25,random_state=4,stratify=targets)

#Dimensionality reduction using PCA
"""Initially it has 4096 features but to reduce it we need to find the optimal number of eigenvectors by 
visualising the plt between explained variance(eigenValue) and principle components"""
pca = PCA(n_components = 100 , whiten = True)   #whiten=True to improve accuracy
pca.fit(train_x)


"""#plot visualisation
plt.figure(1, figsize = (20,20))
plt.plot(pca.explained_variance_, linewidth=2)
plt.xlabel("Number of Eigenvectors / Principle components")
plt.ylabel("Eigenvalue")
plt.show()
#the observed value come out to be something 100"""

train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

#EigenFaces---> are the features remaining after PCA dimensionality reduction
#lets visualise it

"""no_of_eigenfaces = len(pca.components_)   # --->100
eigen_faces = pca.components_.reshape((no_of_eigenfaces,64,64))
fig , subplot = plt.subplots(nrows = 10, ncols = 10, figsize = (15,15))
sub_plot = subplot.flatten()
for i in range(no_of_eigenfaces):
    sub_plot[i].imshow(eigen_faces[i],cmap="gray")
    sub_plot[i].set_xticks([])
    sub_plot[i].set_yticks([])
plt.suptitle("EigenFaces Visualisation")
plt.show()
"""

#Model selection and training

models = [("Logistic Regression" , LogisticRegression()),("Support Vector Machine",SVC()),("Naive Bayes",GaussianNB())]
for model_name , model in models:
    main_model = model
    main_model.fit(train_x, train_y)
    predictions = main_model.predict(test_x)
    print(model_name)
    print("Accuracy:",metrics.accuracy_score(test_y, predictions))
