from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)
# cast to int and float
X = mnist['data'].astype('float64')
t = mnist['target'].astype('int64')
# random permutation of the dataset
random_state = check_random_state(1)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
t = t[permutation]

X = X.reshape((X.shape[0], -1)) #flattens the image into a vector of size 784
bias_vec = np.ones((X.shape[0], 1))
X = np.concatenate((X,bias_vec), axis=1) # adds the 785 bias vec on the right
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2) #splits to train and test
X_train, X_valid, t_train, t_valid = train_test_split(X_train, t_train, test_size=0.25) #splits train to valid and train

# standardize the images each group on its own
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
W = np.random.rand(10,785)*10e-4 # random weights scaled by 5*10^-5

# labels - one hot encoded
n_values = np.max(t) + 1
t_one_hot_train = np.eye(n_values)[t_train]
t_one_hot_valid = np.eye(n_values)[t_valid]

# init vars
adaptive_lr_flag = False
iteration, count = 0,0
correct_perc,last_corr_perc = 0.0,0.0
accuracy_arr, accuracy_arr_train,loss_arr = [],[],[]

while count <= 10:  # break if not improving for 10 rounds
    iteration += 1
    # calc prediction y
    y_denom_train = np.sum(np.exp(W.dot((np.transpose(X_train)))), axis=0)
    y_train = np.exp(W.dot(np.transpose(X_train)))/y_denom_train
    y_predict = np.argmax(y_train, axis=0)
    num_correct_train = np.sum(y_predict == t_train)
    correct_perc_train = 100*(num_correct_train/t_train.shape[0])
    accuracy_arr_train += [correct_perc_train]

    # calc loss - diagonal multiplication for the 2 matrices
    loss = -np.einsum('ij,ji->i', t_one_hot_train, np.log(y_train)).sum(axis=0)
    loss_arr += [loss]

    # loss2 = -np.transpose(t_one_hot.flatten()).dot(test)
    grad_loss = np.transpose(X_train).dot(np.transpose(y_train)-t_one_hot_train)

    # adaptive learning rate according to accuracy on validation if stuck 3 rounds and above 90 accuracy
    if not adaptive_lr_flag:
        eta = 0.0001
    if (count >= 3) & (correct_perc>= 90):
        adaptive_lr_flag = True
        if count >= 4:
            eta = 0.00002
        if count >= 6:
            eta = 0.00001
    W = W - eta*np.transpose(grad_loss) # update Weights

    # Validation check
    y_denom_valid = np.sum(np.exp(W.dot((np.transpose(X_valid)))), axis=0)
    y_valid = np.exp(W.dot(np.transpose(X_valid))/y_denom_valid)
    y_predict_valid = np.argmax(y_valid, axis=0)
    num_correct = np.sum(y_predict_valid == t_valid)
    last_corr_perc = correct_perc if last_corr_perc < correct_perc else last_corr_perc # saves last accuracy if better
    correct_perc = 100*(num_correct/t_valid.shape[0])
    accuracy_arr += [correct_perc]
    count = 1 if correct_perc > last_corr_perc else count + 1  # init counter if improving
    print(f"Valid Accuracy: {correct_perc:.2f}")


# print training accuracy
print(f"Training Accuracy: {correct_perc_train:.2f}")

# Test check
y_denom_test = np.sum(np.exp(W.dot((np.transpose(X_test)))), axis=0)
y_test = np.exp(W.dot(np.transpose(X_test))/y_denom_test)
y_predict_test = np.argmax(y_test, axis=0)
num_correct_test = np.sum(y_predict_test == t_test)
correct_perc_test = 100*(num_correct_test/t_test.shape[0])
print(f"Test Accuracy: {correct_perc_test:.2f}")

iteration = np.linspace(0, iteration, iteration)
plt.figure(1)
plt.plot(iteration, accuracy_arr_train, 'g', label='Training') # plotting t, b separately
plt.plot(iteration, accuracy_arr, 'b', label='Validation') # plotting t, b separately
plt.plot(iteration, correct_perc_test*np.ones_like(accuracy_arr), 'r', label='Test') # plotting t, b separately
plt.legend()
plt.ylabel("Accuracy %")
plt.suptitle("Accuracy (iterations) [%]")
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-')
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.savefig('/Users/danmalka/Documents/לימודים/שנה ג/סמסטר ב/Machine Learning/MNIST HW 4/Accuracy.pdf')

plt.figure(2)
plt.plot(iteration, loss_arr, 'r', label='Loss') # plotting t, c separately
plt.legend()
plt.suptitle("Loss (iterations)")
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-')
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.savefig('/Users/danmalka/Documents/לימודים/שנה ג/סמסטר ב/Machine Learning/MNIST HW 4/Loss.pdf')
plt.show()