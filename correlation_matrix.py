import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, learning_curve
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import statsmodels.api as sm
df = pd.read_csv('Test.csv')
y= df.loc[:, df.columns == 'RWC']
print(y)

X = df.loc[:, df.columns != 'RWC']
print(X)
##### Univariate ANalysis #########

X.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
                       xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout() 
plt.show()
##################################
# Normalize the data
from sklearn import preprocessing
data = pd.DataFrame(preprocessing.scale(X),columns = X.columns) 

##### Univariate ANalysis #########

data.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
                       xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout()
plt.show()
##################################
########### Correlation Matrix OPtion 1 ###################
print(data)
corr = data.corr()
print(corr)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()
#############################################################
############ Correlation Matrix Option 2 #####################
f, ax = plt.subplots(figsize=(10, 6))
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                                  linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('RGB Polarization and Texture Correlation', fontsize=14)
plt.show()
################################################################
############ Pair wise scatter plot
#cols = ["Sb_std", "Sb_mean", "Sg_std", "Sg_mean", "Sr_std", "Sr_mean", "b_diss", "b_energy", "g_diss", "g_energy", "r_diss", "r_energy"]
cols = ["P1_std_b", "P1_mean_b", "P1_std_g", "P1_mean_g", "P1_std_r", "P1_mean_r", "diss_b", "energy_b", "diss_g", "energy_g", "diss_r", "energy_r"]
pp = sns.pairplot(X[cols], size=1.8, aspect=1.8,
                                    plot_kws=dict(edgecolor="k", linewidth=0.5),
                                    diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=11)
plt.show()
#############################################################
pca = PCA(n_components=2)
X = pca.fit_transform(data)


pcsummary = pd.DataFrame(pca.components_,columns=data.columns,index = ['PC-1','PC-2'])
print("PCA", pca.explained_variance_ratio_)
f = open('pca.tex', 'w')
f.write(pcsummary.to_latex())
f.close()
############# Linear Regression 1 #################
regr = LinearRegression()
regr.fit(X, y)
y_pred = regr.predict(X)
#print(y)
# Calculate and print classifier metrics
R2 = r2_score(y, y_pred, multioutput='variance_weighted' )
print("r2: ", R2)
n = len(y)
p = X.shape[1]
r2_adjusted = 1-(1-R2)*(n-1)/(n-p-1)
print("r2 adjusted", r2_adjusted)
print("root mean squared: ", mean_squared_error(y, y_pred))
#############Linear Regression 2 #######################
X_fit = sm.add_constant(X)
results = sm.OLS(y, X_fit).fit()
print(results.summary())
f = open('myreg.tex', 'w')
f.write(results.summary().as_latex())
f.close()
############################################################
############# 3D Plot of PC1 and PC2 and RWC ####################
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(X[:,0], X[:,1], y, cmap='Greens')
x_surf, y_surf = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 100),np.linspace(X[:,1].min(), X[:,1].max(), 100))
c = regr.intercept_
b = regr.coef_[0][0]
a = regr.coef_[0][1]
Z = c + a*x_surf + b*y_surf 
print(f"Intercept: {c}")
print(f"Coef 1: {b}")
print(f"Coef 2: {a}")
ax.set_xlabel('Component 1', labelpad=5)
ax.set_ylabel('Component 2', labelpad=5)
ax.set_zlabel('RWC')
ax.plot_surface(x_surf,y_surf, Z,  alpha=0.1)
plt.show()
#################################################################
