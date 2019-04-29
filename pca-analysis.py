from __future__ import division
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


######################
# Regression imports #
from sklearn import linear_model
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, learning_curve
######################
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
def PolynomialRegression(degree=2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


########################
###### PCA IMports #####
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
#######################
### Texture imports ###
from sklearn import svm
from sklearn.feature_extraction import image
from skimage.feature import greycomatrix, greycoprops
#######################
#######################
#Specular RWC samples
# (imgs)     1_1 -> 99.0756
# (imgs2)    31  -> 97.6573
# (imgs3)    3^4 -> 96.6949
# (imgs4)   2+1 -> 88.8809
# (imgs5)   2+2 -> 92.4017
# (imgs6)   3+1 -> 95.4651
#######################


# we should ask for path name of directory and if not use the current one

#for img_file in os.listdir(pathname):
#    print img_file
#    if img_file == 'H.png':
#   
# input directory name and image name
# returns list of flux values for polarization filtered image
def img_to_flux(P_img):
    P = P_img.ravel().astype(np.int16)
    return P

# input two arrays of flux values
# output numpy array of stokes values
def calc_stokes(P1, P2):
    S = []
    for P1_px,P2_px in zip(P1,P2):
        denom = P1_px + P2_px
        # Filter out if values never change from 0
        #if denom == 0:
        #    continue
        # Filter out if values are ever higher than 225
        if P1_px == 0 or P2_px == 0:
            continue
        elif P1_px > 254 or P2_px > 254:
            continue
    #    elif P1 < 25 or P2 < 25:
    #        continue
        else:
            calc = (P1_px - P2_px) / (P1_px + P2_px)
            S.append(calc)
    return np.array(S)

# Show image using CV2
def show_image(img):
    image_path = os.path.join(pathname, img)
    raw_img = cv2.imread(image_path, 0)
    cv2.imshow('image', raw_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.waitKey(1)
#H = [px for px in H if px < 225 and px != 0]

# TODO add printing yes no option
def stats_stokes(S):
    printer = False
    maximum = max(S)
    minimum = min(S)
    mean = S.mean()
    std = S.std()

    if printer == True:
        print("MAX: ", maximum)
        print("MIN: ", minimum)
        print("MEAN: ", mean)
        print("STD: ", std)

    stats = {}
    stats["mean"] = mean
    stats["std"] = std
    stats["min"] = minimum
    stats["max"] = maximum
    return stats

def plot_histogram(S):
    plt.hist(S, bins=255)
    plt.show()

def stokes_analysis(P1_img, P2_img):
    # Options
    show_images = False
    calc_stats = True
    plot_hist = False

    # Convert image pixel intensity to flux values
    P1 = img_to_flux(P1_img)
    P2 = img_to_flux(P2_img)

    # Calculate the Stokes parameter
    S = calc_stokes(P1, P2)
    results = {}
    results["S"] = S

    if show_images == True:
        show_image(P1_img)
        plot_histogram(img_to_flux(P1_img))
        show_image(P2_img)
        plot_histogram(img_to_flux(P1_img))
    if plot_hist == True:
        plot_histogram(S)
    if calc_stats == True:
        stats = stats_stokes(S)
        results["stats"] = stats
    return results
#####################
## RWC Plotting   ##
#####################
# Plot a couple RWCs against S1 polarization mean and std
# X is now X_stds since its results were more promising
X = [0.4278,0.3074, 0.3332, 0.5124, 0.4721, 0.4320]
X_means = [-0.2640, -0.1660, 0.1327, -0.04508, 0.1063, 0.1080]
y = [98.4379, 97.6573, 96.6949, 88.8809, 92.4017, 95.4651]
#plt.scatter(X, y)
#plt.show()

#######################
## Linear regression ##
#######################
# Question: how do we group X and y
def read_dat_file(pathname):
    data = []
    with open(pathname) as dat_file:
        for value in dat_file:
            data.append(value.rstrip('\n'))
    return float(data[0])
#reshape since it is only one feature at the moment
def linear_analysis(X, y):
    # TODO TRY AND REMOVE THIS LINE
    data = np.array(X)
    # Scale each feature column to have zero mean and 1 STD
    
    
    
    
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    #X = normalize(X)



    #print("####LINEAR ANALYSIS#####")
    #print(X.mean(axis = 0))
    #print(X.std(axis = 0))
#    y = np.array(y)
    # ONLY NEED THIS IF THERE IS ONE FEATURE
    #pca = PCA(n_components=2).fit(X_train)
    pca = PCA(n_components=2)
    #X_train_pca = pca.transform(X_train)
    #X_test_pca = pca.transform(X_test)

    X_pca = pca.fit_transform(X)
    #X = X.reshape(-1,1)
    print("PCA", pca.explained_variance_ratio_)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=2)
    #steps = [
    #        ('scale', StandardScaler()),
    #        ('pca', PCA()),
    #        ('estimator', LinearRegression())
    #]
    #pipe = Pipeline(steps)
    #pca = pipe.set_params()
    #pipe.fit(X_train, y_train)
    #TODO: Add StandardScaler
    #TODO: Make a pipeline overall for imaging process
############################################
        # X_axis = np.linspace(0,100,1).T
    #pca = PCA(n_components=2).fit(X_train)
#    print(X_pca.explained_variance_ratio_)
    #print(X_pca)
    #X_pca = X_pca.ravel()
    # This is only needed if there is one sample
    # TODO: Add cross fold validation
    #X_pca = X_pca.reshape(-1,1)
    loo = LeaveOneOut()
    regr = linear_model.LinearRegression()
    #regr.fit(X_train_pca, y_train)
    folds = len(X)
    scores = cross_val_score(regr, X, y, cv=5)
    
    print(scores)
    print("Accurcy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #regr.fit(X_train, y_train)
    #y_pred = regr.predict(X_test)
    regr.fit(X_pca, y)
    y_pred = regr.predict(X_pca)
    #print(y_pred)
    #print(y)
    # Calculate and print classifier metrics
#    print(classification_report(y_test, y_pred, target_names=target_names))
    R2 = r2_score(y, y_pred, multioutput='variance_weighted' )
    print("r2: ", R2)
    n = len(y)
    print("SHAPE", X.shape)
    p = X_pca.shape[1]
    print(n)
    print(p)
    r2_adjusted = 1-(1-R2)*(n-1)/(n-p-1)
    #print("explained_variance_score: ", explained_variance_score(y, y_pred))
    print("r2 adjusted", r2_adjusted)
    print("root mean squared: ", mean_squared_error(y, y_pred))

    plt.title('Linear Regression For Relative Water Content')
    plt.xlabel('First Principal Component')
    plt.ylabel('Relative Water Content')
    plt.scatter(X_pca[:,0], y)
    plt.scatter(X_pca[:,0].reshape(-1, 1), y_pred, marker='^')

    plt.show()


######################################################
####OPTION 1 for validation curve#######################
#######################################################
#    from sklearn.model_selection import validation_curve
#    degree = np.arange(0, 9)
#    train_score, val_score = validation_curve(regr, X_pca, y, 'polynomialfeatures__degree', degree, cv=5)
#    
#    plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
#    plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
#    plt.legend(loc='best')
#    plt.ylim(-0.1, 1.1)
#    plt.xlabel('degree')
#    plt.ylabel('score');
#    plt.show()
###########################################
######## OPTION 2 for learning curve ######
############################################
#    plt.figure()
#    plt.title("title")
#    plt.ylim(-10,2)
#    plt.xlabel("Training examples")
#    plt.ylabel("Score")
#    train_sizes, train_scores, test_scores = learning_curve(regr, X_pca, y, cv=5,train_sizes=np.linspace(.1, 1.0, 5))
#    train_scores_mean = np.mean(train_scores, axis=1)
#    train_scores_std = np.std(train_scores, axis=1)
#    test_scores_mean = np.mean(test_scores, axis=1)
#    test_scores_std = np.std(test_scores, axis=1)
#    plt.grid()
#
#    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#    train_scores_mean + train_scores_std, alpha=0.1,
#    color="r")
#    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#    test_scores_mean + test_scores_std, alpha=0.1, color="g")
#    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#    label="Training score")
#    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#    label="Cross-validation score")
#    plt.legend(loc="best")
#    plt.show()

    ############ 3D plot ################
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(df2['Price'],df2['AdSpends'],df2['Sales'],c='blue', marker='o', alpha=0.5)
    #ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='None', alpha=0.01)
    #ax.set_xlabel('Price')
    #ax.set_ylabel('AdSpends')
    #ax.set_zlabel('Sales')
    #plt.show()
###################
# Texture Analysis #
####################
def texture_analysis(img):
    img_samples = image.extract_patches_2d(img, (75, 75), 20, 1)
    texture = []
    for sample in img_samples:
         try:
         # Calculate texture features for a given sample
             relationships = [0, np.pi/4, np.pi/2, 3*np.pi/4]
             glcm = greycomatrix(sample, [1], relationships, 256, symmetric=True, normed=True)
             metrics = ['dissimilarity', 'contrast', 'correlation', 'energy']
             diss, contrast, corr, energy = [greycoprops(glcm, metric)[0, 0] for metric in metrics]

             texture.append([diss, contrast, corr, energy])
         except ValueError:
             print("Error in extracting the texture features")
    #Xt = np.array(texture)
    #print(Xt)
    data = np.array(texture)
    diss = np.mean(data[:,0])
    contrast = np.mean(data[:,1])
    energy = np.mean(data[:,3])
    return [diss, energy]
####################
def main():
    raw_data_path = os.path.join(os.getcwd(), "raw_data")
    raw_data = os.listdir(raw_data_path)
    datasets  = [f for f in raw_data if not f.startswith('.')]
    #print(datasets)
    #print(raw_data_path)
    options = {}
    options["bgr"] = True
    options["texture"] = True
    options["stokes"] = True

    P1_img_name = 'H.png'
    P2_img_name = 'V.png'
#    imgs = ['H.png', 'V.png']
    X = []
    y = []
    headers = ["RWC", "P1_std_b", "P1_mean_b", "P1_std_g", "P1_mean_g", "P1_std_r", "P1_mean_r", "diss_b", "energy_b", "diss_g", "energy_g", "diss_r", "energy_r"]
#    headers = ["Sb_std", "Sb_mean", "Sg_std", "Sg_mean", "Sr_std", "Sr_mean"] 
    with open ('Test.csv', 'a') as f:
        np.savetxt(f, [headers], delimiter=',', fmt="%s")
    f.close()
    for directory in datasets:
        pathname = os.path.join(raw_data_path, directory)

        # RWC Infromation
        dat_file_path = os.path.join(pathname, "rwc.dat")
        rwc = read_dat_file(dat_file_path)
        if options["bgr"] == True:
            COLOR = 1
        else:
            COLOR = 0

        P1_img = cv2.imread(os.path.join(pathname, P1_img_name), COLOR)
        P2_img = cv2.imread(os.path.join(pathname, P2_img_name), COLOR)

        Xs = []
        Xt = []
        headers = []
        # Texture Analysis
        if options["texture"] == True and options["bgr"] == True:
            T_bgr = cv2.split(P1_img)
            #TODO: add support for both images to be analyzed
            T = [texture_analysis(P1_img) for P1_img in T_bgr]
            #blue = cv2.split(P1_img)[0]
            #t_bgr = zip(cv2.split(P1_img), cv2.split(P2_img))
            for t in T:
                Xt.extend(t)
        elif options["texture"] == True:
            Xt.extend(texture_analysis(P1_img))
            Xt.extend(texture_analysis(P2_img))
            #Xt.append(texture_analysis(P2_img))
            #print(texture)
        # Stokes analysis
        if options["stokes"] == True and options["bgr"] == True:
            P_bgr = zip(cv2.split(P1_img), cv2.split(P2_img))
            S = [stokes_analysis(P1, P2) for P1,P2 in P_bgr]
            for s in S:
                Xs.extend([s["stats"]["std"],s["stats"]["mean"]])
        elif options["stokes"] == True:
            S = stokes_analysis(P1_img, P2_img)
            Xs.extend([S["stats"]["std"],S["stats"]["mean"]])
        #print(S)
        print("#")
        # print("#", end='')
        #print(data["stats"])
        #print("RWC: {}".format(data["rwc"]))
        #X.append(Xs)
        #X.append(Xt)
        #X.append([S[0]["stats"]["std"], S[0]["stats"]["mean"], S[1]["stats"]["std"], S[1]["stats"]["mean"]])
        #X_tmp = []
        # Extend feature arrays
        #Xs.extend(Xt)
        X.append(Xs + Xt)
        #X.append([data["stats"]["mean"]])
        y.append(rwc)
        list_features = [rwc] + Xs + Xt
        print(list_features)
        with open ('Test.csv', 'a') as f:
            np.savetxt(f, [list_features], delimiter=',', fmt="%g")
        f.close()
    # Linear Regression
    #print("X: {}".format(X))
    #print("y: {}".format(y))
    #X = normalize(X)
#    pca = sklearnPCA(n_components=2)
 #   X = pca.fit_transform(X)
#    print(Xs)
#    print(Xt)
    #X = [np.append(xs, xt) for xs, xt in zip(Xs, Xt)]
    print(X)
    linear_analysis(X,y)

if __name__ ==  '__main__':
    main()
