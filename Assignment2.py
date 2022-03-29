import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

##(c)
data = pd.read_csv('songs.csv')
#(I)
remove_features = ['Artist Name','Track Name','key','mode','time_signature','instrumentalness']
data = data.drop(remove_features,axis=1)
##(II)
data = data[(data['Class']== 5) | (data['Class']== 9)]
data['Class'].replace([5,9],[1,-1],inplace=True)
data = data.reset_index(drop=True)
#(III)
data = data.dropna()
data.isnull().sum()
#(IV)
X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[:,0:-1], data.iloc[:,-1], test_size=0.3, random_state=23)
#(V)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#(VI)
X_train = pd.DataFrame(X_train, 
             columns=['Popularity', 
                      'danceability', 
                      'energy', 
                      'loudness', 
                      'speechiness', 
                      'acousticness', 
                      'liveness', 
                      'valence', 
                      'tempo', 
                      'duration_in min/ms'])

X_test = pd.DataFrame(X_test, 
             columns=['Popularity', 
                      'danceability', 
                      'energy', 
                      'loudness', 
                      'speechiness', 
                      'acousticness', 
                      'liveness', 
                      'valence', 
                      'tempo', 
                      'duration_in min/ms'])

Y_train = pd.DataFrame(Y_train, 
             columns=['Class']) 

Y_test = pd.DataFrame(Y_test, 
             columns=['Class']) 

X_train_first = X_train.iloc[0,0:3]
X_train_last = X_train.iloc[-1,0:3]
print("X_train first row is ")
print(X_train_first)
print("\nX_train last row is")
print(X_train_last)

X_test_first = X_test.iloc[0,0:3]
X_test_last = X_test.iloc[-1,0:3]
print("\nX_test is first row is ")
print(X_test_first)
print("\nX_test last row is")
print(X_test_last)


Y_train_first = Y_train.iloc[0]
Y_train_last = Y_train.iloc[-1]
print("\nY_train first row is ")
print(Y_train_first)
print("\nY_train last row is")
print(Y_train_last)

Y_test_first = Y_test.iloc[0]
Y_test_last = Y_test.iloc[-1]
print("\nY_test first row is ")
print(Y_test_first)
print("\nY_test last row is")
print(Y_test_last)

##(e)
def reg_log_loss(W, C, X, y):
    c = W[0]#1x1
    w_T = W[1:].reshape(1,X.shape[1]) 
    X = X.values
    Y = y.values
    sum = 0
    
    
    for i in range(0,2720):
        x = X[i].reshape(X.shape[1],1)
        y = Y[i]
        wx = np.dot(w_T,x)
        exponent = -1*y*(wx+c)
        sum = sum + np.logaddexp(0,exponent)
   
    loss = 0.5*(np.linalg.norm(w_T, ord = 2 )**2)+C*sum
    return loss[0]

w = 0.35 * np.ones(X_train.shape[1])
c=1.2
W = np.insert(w, 0, c)
loss = reg_log_loss(W, 0.001, X_train, Y_train)
print(loss[0])


##(f)
#(I)
def reg_log_fit(X,y,C):
    w = 0.1 * np.ones(X_train.shape[1])
    W0 = np.insert(w,0, -1.4)
    g = lambda W:  reg_log_loss(W, C, X, y)
    optimize = minimize(g, W0, method="Nelder-Mead", tol=1e-6, options = {'maxfev': 7115})
    return optimize

result1 = reg_log_fit(X_train, Y_train, 0.4) 
optimal_W = result1.x

def y_pred(W, X):
    c = W[0]
    w = W[1:].reshape(1,X.shape[1]) 
    X = X.values
    Y_predict = []
    
    for i in range(0,X.shape[0]):
        x = X[i]
        wx = np.dot(w,x)
        exponent = -1*(wx+c)
        y_predict = 1/(1 + np.exp(exponent)) #p(y=1|x):hiphop
        Y_predict.append(y_predict)
    return Y_predict   

Y_train_predict = y_pred(optimal_W, X_train)
Y_test_predict = y_pred(optimal_W,X_test)

train_loss = log_loss(Y_train, Y_train_predict)
test_loss = log_loss(Y_test, Y_test_predict)

print("Train loss is {}".format(train_loss))
print("Test loss is {}".format(test_loss))

#(II)
model = LogisticRegression(penalty='l2',tol=1e-6,C=1,solver='liblinear')
y_train =  Y_train.values.ravel()
model.fit(X_train,y_train)

W = np.insert(model.coef_, 0, model.intercept_)
print(W)

train_loss = log_loss(Y_train, y_pred(W, X_train)) 
test_loss = log_loss(Y_test, y_pred(W, X_test))

print("Train loss is {}".format(train_loss))
print("Test loss is {}".format(test_loss))


##(g)
Cs = np.linspace(0.001, 0.2, num=100) #generate the list of C values
coefficients = []
for c in Cs:
    l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=c)
    l1_model.fit(X_train,y_train)
    coef = l1_model.coef_[0] #to aviod double bracket [[]]
    coefficients.append(coef)

coefficients = pd.DataFrame(coefficients)
colors = ['red', 'brown', 'green', 'blue', 'orange', 'pink', 'purple', 'grey', 'black', 'y']
lables = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'liveness', 'valence', 'tempo', 'duration_in min/ms']

for i in range(10):
    plt.plot(np.log(Cs), coefficients.iloc[:,i], color = colors[i], label = lables[i])
    
plt.legend(bbox_to_anchor = (1.05, 1))
plt.xlabel("log(C)")
plt.ylabel("coefficient")


##(h)
Cs = np.linspace(0.0001, 0.8, num=25)
small_x_train = X_train.iloc[:544]
small_y_train = Y_train.iloc[:544]
small_y_train = small_y_train.reset_index(drop=True)

avg_loss = []
for c in Cs:
    loss_sum = 0
    _model = LogisticRegression(penalty='l1',solver='liblinear', C=c)
    for i in range(544):
        x_tr = small_x_train.drop(index = i).values
        y_tr = small_y_train.drop(index = i).values.ravel()
        
        x_ts = small_x_train.iloc[i].values.reshape(1, -1)
        y_ts = small_y_train.iloc[i].values.ravel()  
        
        _model.fit(x_tr,y_tr)
        y_pred = _model.predict_proba(x_ts)
        loss_ = log_loss(y_ts, y_pred, labels = (-1,1))
        loss_sum = loss_sum + loss_
    avg = loss_sum / 544
    avg_loss.append(avg)

plt.plot(Cs, avg_loss)
plt.xlabel("C")                                    
plt.ylabel("loss")

for i in range(len(Cs)):
    loss = avg_loss[i]
    if loss == min(avg_loss):
        break
#print(i)
#avg_loss[12]
#Cs[12]



### Question2
import numpy as np
import pandas as pd # not really needed, only for preference
import matplotlib.pyplot as plt 
from util import *

##(a)
X = pd.read_csv('PerceptronX.csv', header=None)
Y = pd.read_csv('Perceptrony.csv',header=None)
x = X.values
y = Y.values

w0 = np.zeros(3) #3,
w0.ravel()

def perceptron(X, y, max_iter=100):
    np.random.seed(1)
    w = np.zeros(3)
    nmb_iter = 0
    converged = False
    while converged == False:
        converged = True
        nmb_iter = nmb_iter + 1
        misscl = [] 
        
        if nmb_iter > max_iter:
            break
        
        for i in range(0,82):
            x = X.iloc[i].values
            y = Y.iloc[i].values
            xy = np.insert(y, 0, x)
            wx = np.dot(w,x)
            
            if y*wx <= 0:
                misscl.append(xy)
                converged = False
            
        if converged == False:
            ind = np.random.choice(len(misscl)) #pick a random index
            random_row = misscl[ind] # pick a corresponding row
            x_ = random_row[:-1]
            y_ = random_row[-1]
            w = w + y_*x_
        
    return w, nmb_iter-1    

w, nmb_iter = perceptron(X, Y, max_iter=100)

fig, ax = plt.subplots()
plot_perceptron(ax, x, y, w) 
ax.set_title(f"w={w}, iterations={nmb_iter}")
plt.show()

##(b)
Gram = X@X.T

def dual_perceptron(X, y, max_iter=100):
    np.random.seed(1)
    alpha = alpha = np.zeros(82)
    nmb_iter = 0
    converged = False
    while converged == False:
        converged = True
        nmb_iter = nmb_iter + 1
        misscl_i = [] 
        
        if nmb_iter > max_iter:
            break
        
        for i in range(0,82):
            sum = 0 
            x_i = X.iloc[i].values
            y_i = Y.iloc[i].values
            
            for j in range(0,82):
                y_j = Y.iloc[j].values
                a_j = alpha[j]
                sum = sum + a_j*y_j*Gram.loc[j,i]
            
            condition = y_i * sum
            if condition <= 0:
                misscl_i.append(i)
                converged = False
  
        if converged == False:
            ind = np.random.choice(len(misscl_i)) #pick a random index
            random_row = misscl_i[ind] # pick a corresponding row
            alpha[random_row] = alpha[random_row]+1
    return alpha, nmb_iter-1

alpha, t = dual_perceptron(X, Y, max_iter=100)

w_ = np.zeros(3)
for i in range(0,82):
    a_i = alpha[i]
    x_i = X.iloc[i].values
    y_i = Y.iloc[i].values
    w_ = w_ + a_i*y_i*x_i
W = w

fig, ax = plt.subplots()
plot_perceptron(ax, x, y, W) 
ax.set_title(f"w={w}, iterations={t}")
plt.show()

i = np.linspace(1,82,82)
plt.plot(i,alpha)
plt.xlabel('x')
plt.ylabel('alpha')
plt.show()

##(c)
def rperceptron(X, y, max_iter=100):
    np.random.seed(1)
    w = np.zeros(3)
    I = np.zeros(82)
    nmb_iter = 0
    converged = False
    
    while converged == False:
        converged = True
        nmb_iter = nmb_iter + 1
        misscl = [] 
        
        if nmb_iter > max_iter:
            break
        
        for i in range(0,82):
            x = X.iloc[i].values
            y = Y.iloc[i].values
            xy = np.insert(y, 0, x)
            wx = np.dot(w,x)
        
            if y*wx + 2*I[i] <= 0:
                misscl.append(i)
                converged = False
                
        if converged == False:
            ind = np.random.choice(len(misscl)) #pick a random index of misscl
            random_ind = misscl[ind] # find the random index in terms of i
            I[random_ind] = 1      

            x_ = X.iloc[random_ind].values
            y_ = Y.iloc[random_ind].values
    
            w = w + y_*x_
              
    return w, nmb_iter-1         

r_w, r_nmb_iter = rperceptron(X, Y, max_iter=100)

fig, ax = plt.subplots()
plot_perceptron(ax, x, y, r_w) 
ax.set_title(f"w={r_w}, iterations={r_nmb_iter}")
plt.show()


##(d)
def rdual_perceptron(X, y, max_iter=100):
    np.random.seed(1)
    alpha = alpha = np.zeros(82)
    nmb_iter = 0
    I = np.zeros(82)
    converged = False
    while converged == False:
        converged = True
        nmb_iter = nmb_iter + 1
        misscl_i = [] 
        
        if nmb_iter > max_iter:
            break
        
        for i in range(0,82):
            sum = 0 
            x_i = X.iloc[i].values
            y_i = Y.iloc[i].values
            
            for j in range(0,82):
                y_j = Y.iloc[j].values
                a_j = alpha[j]
                sum = sum + a_j*y_j*Gram.loc[j,i]
            
            condition = y_i * sum + 2*I[i]
            if condition <= 0:
                misscl_i.append(i)
                converged = False
        
        if converged == False:
            ind = np.random.choice(len(misscl_i)) #pick a random index of misscl_i
            random_ind = misscl_i[ind] # pick a random index of data
            I[random_ind] = 1 
            alpha[random_ind] = alpha[random_ind]+1
            
    return alpha, nmb_iter-1

r_alpha, rnmb_iter = rdual_perceptron(X, Y, max_iter=100)
w_ = np.zeros(3)
for i in range(0,82):
    a_i = r_alpha[i]
    x_i = X.iloc[i].values
    y_i = Y.iloc[i].values
    w_ = w_ + a_i*y_i*x_i
W = w_

fig, ax = plt.subplots()
plot_perceptron(ax, x, y, W) 
ax.set_title(f"w={W}, iterations={rnmb_iter}")
plt.show()

i = np.linspace(1,82,82)
plt.plot(i,r_alpha)
plt.xlabel('x')
plt.ylabel('alpha')
plt.show()

