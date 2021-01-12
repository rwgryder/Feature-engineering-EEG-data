from balance_dataset import input_output
import numpy as np
import pandas as pd
import genetic
import seaborn as sns
import matplotlib.pyplot as plt
import mutual_info
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler

# retrieve full data set
DATASET_FOLDER = 'data/2000ms'

FILE_LIST = ['wav75_ex1_.csv','wav75_ex2_.csv','wav75_ex3_.csv','wav75_ex4_.csv','wav75_ex5_.csv','wav75_ex6_.csv','wav75_ex7_.csv','wav75_ex8_.csv']
data = input_output.read_dataset_list(DATASET_FOLDER, FILE_LIST)

column_id = list(range(0,75))
column_id.append(76)

X = data[data.columns[column_id]].values
y = data[data.columns[75]].values

# operations in Genetic Program
function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log']
                #'abs', 'neg', 'inv',
                #'max', 'min']

# The Genetic Program class
gp = genetic.SymbolicTransformer(generations=20, population_size=1000,
                         tournament_size = 20, n_components=10, hall_of_fame = 1000,
                         function_set=function_set,
                         max_samples=1, verbose=1, n_jobs=1,
                         p_crossover = 0.85, p_subtree_mutation = 0.15, p_point_mutation = 0, p_hoist_mutation = 0,
                         metric='mutual information', stopping_criteria = 10^6)

# A comparison to PCA can be easily implemented here
#pca = PCA(n_components = 10)

# Models to fit on engineered features
nb = GaussianNB()
dt = tree.DecisionTreeClassifier()
sv = svm.SVC(kernel = 'rbf')
knn = KNeighborsClassifier(n_neighbors = 5)
mlp = MLPClassifier(hidden_layer_sizes=15, activation='relu')


# outputs train and test data that is randomly split; training classes are randomly balanced
def processed_data(X, y, split = 0.3):
    
    # randomly split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= split, shuffle = True)  
    
    y_test = X_test[:,75]
    X_test = X_test[:,0:75]
    X_train = X_train[:,0:75]
    
    # randomly balance training data
    ratio = 1  
    rus = RandomUnderSampler(sampling_strategy = ratio)
    X_train, y_train = rus.fit_resample(X_train, y_train)   
    
    return X_train, y_train, X_test, y_test


# outputs model performance metrics on engineered features of test data
def metrics(X_train, y_train, X_test, y_test, clf):
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    try:
        auc = roc_auc_score(y_test, y_pred)
    except:
        auc = 0
    
    if auc < 0.5:
        auc = 1 - auc  
    
    recall = tp / (tp+fn) 
    spec = precision_recall_fscore_support(y_test, y_pred)[0][0]
    prec = precision_recall_fscore_support(y_test, y_pred)[1][0]
    f1 = precision_recall_fscore_support(y_test, y_pred)[2][0]
    acc = accuracy_score(y_test, y_pred)
    
    return auc, spec, prec, recall, f1, acc


# organizes results for plotting
def get_results(nb_auc, nb_spec, nb_prec, nb_recall, nb_f1, nb_acc, dt_auc, dt_spec, dt_prec, dt_recall, dt_f1, dt_acc, sv_auc, sv_spec, sv_prec, sv_recall, sv_f1, sv_acc, knn_auc, knn_spec, knn_prec, knn_recall, knn_f1, knn_acc, mlp_auc, mlp_spec, mlp_prec, mlp_recall, mlp_f1, mlp_acc):
    
    nb_model = 'Naive Bayes'
    nb_res = {'AUC': nb_auc, 'Recall': nb_recall, 'Specificity': nb_spec, 'Precision': nb_prec, 'F1': nb_f1, 
              'Accuracy': nb_acc, 'Model': [nb_model]*10}
    nb_results = pd.DataFrame(data = nb_res)
    
    dt_model = 'Decision Tree'
    dt_res = {'AUC': dt_auc, 'Recall': dt_recall, 'Specificity': dt_spec, 'Precision': dt_prec, 'F1': dt_f1, 
              'Accuracy': dt_acc, 'Model': [dt_model]*10}
    dt_results = pd.DataFrame(data = dt_res)
    
    sv_model = 'SVM'
    sv_res = {'AUC': sv_auc, 'Recall': sv_recall, 'Specificity': sv_spec, 'Precision': sv_prec, 'F1': sv_f1, 
              'Accuracy': sv_acc, 'Model': [sv_model]*10}
    sv_results = pd.DataFrame(data = sv_res)
    
    knn_model = 'K-nn'
    knn_res = {'AUC': knn_auc, 'Recall': knn_recall, 'Specificity': knn_spec, 'Precision': knn_prec, 'F1': knn_f1, 
              'Accuracy': knn_acc, 'Model': [knn_model]*10}
    knn_results = pd.DataFrame(data = knn_res)
    
    mlp_model = 'Multi-layer Perceptron'
    mlp_res = {'AUC': mlp_auc, 'Recall': mlp_recall, 'Specificity': mlp_spec, 'Precision': mlp_prec, 'F1': mlp_f1, 
              'Accuracy': mlp_acc, 'Model': [mlp_model]*10}
    mlp_results = pd.DataFrame(data = mlp_res)
    
    all_results = pd.concat([sv_results, nb_results, knn_results, dt_results, mlp_results], axis=0)
    
    return(all_results)
    

#Mi = np.zeros(10)

nb_auc = np.zeros(10)
nb_acc = np.zeros(10)
nb_recall = np.zeros(10)
nb_spec = np.zeros(10)
nb_prec = np.zeros(10)
nb_f1 = np.zeros(10)

dt_auc = np.zeros(10)
dt_acc = np.zeros(10)
dt_recall = np.zeros(10)
dt_spec = np.zeros(10)
dt_prec = np.zeros(10)
dt_f1 = np.zeros(10)

sv_auc = np.zeros(10)
sv_acc = np.zeros(10)
sv_recall = np.zeros(10)
sv_spec = np.zeros(10)
sv_prec = np.zeros(10)
sv_f1 = np.zeros(10)

knn_auc = np.zeros(10)
knn_acc = np.zeros(10)
knn_recall = np.zeros(10)
knn_spec = np.zeros(10)
knn_prec = np.zeros(10)
knn_f1 = np.zeros(10)

mlp_auc = np.zeros(10)
mlp_acc = np.zeros(10)
mlp_recall = np.zeros(10)
mlp_spec = np.zeros(10)
mlp_prec = np.zeros(10)
mlp_f1 = np.zeros(10)

# main loop for 10 test/train splits, may take a few minutes!!!
for i in range(0,10):
    
    # process data 
    X_train, y_train, X_test, y_test = processed_data(X, y)
    
    # run feature selector
    gp.fit(X_train, y_train)
    gp_X_train = gp.transform(X_train)
    gp_X_test = gp.transform(X_test)
    
    # PCA
    #pca.fit(X_train)
    #pca_X_train = pca.transform(X_train)
    #pca_X_test = pca.transform(X_test)
    
    
    #mi = np.zeros(10)
    #for k in range(0,10):
    #    mi[k] = mutual_info.mutual_information_2d(gp_X_train[:,k],y_train)
    #Mi[i] = np.mean(mi)
    
    # get 5 metrics from 5 models
    nb_auc[i], nb_spec[i], nb_prec[i], nb_recall[i], nb_f1[i], nb_acc[i] = metrics(gp_X_train, y_train, gp_X_test, y_test, nb)
    dt_auc[i], dt_spec[i], dt_prec[i], dt_recall[i], dt_f1[i], dt_acc[i] = metrics(gp_X_train, y_train, gp_X_test, y_test, dt)
    sv_auc[i], sv_spec[i], sv_prec[i], sv_recall[i], sv_f1[i], sv_acc[i] = metrics(gp_X_train, y_train, gp_X_test, y_test, sv)
    knn_auc[i], knn_spec[i], knn_prec[i], knn_recall[i], knn_f1[i], knn_acc[i] = metrics(gp_X_train, y_train, gp_X_test, y_test, knn)
    mlp_auc[i], mlp_spec[i], mlp_prec[i], mlp_recall[i], mlp_f1[i], mlp_acc[i] = metrics(gp_X_train, y_train, gp_X_test, y_test, mlp)
    

# put together all results
all_results = get_results(nb_auc, nb_spec, nb_prec, nb_recall, nb_f1, nb_acc, dt_auc, dt_spec, dt_prec, dt_recall, dt_f1, dt_acc, sv_auc, sv_spec, sv_prec, sv_recall, sv_f1, sv_acc, knn_auc, knn_spec, knn_prec, knn_recall, knn_f1, knn_acc, mlp_auc, mlp_spec, mlp_prec, mlp_recall, mlp_f1, mlp_acc)

   
# Boxplot of accuracies
all_results['Model'][all_results['Model'] == 'Decision Tree'] = 'DT'
all_results['Model'][all_results['Model'] == 'Naive Bayes'] = 'NB'
all_results['Model'][all_results['Model'] == 'Multi-layer Perceptron'] = 'MLP'
sns.set_style("whitegrid")
sns.boxplot('Model','Accuracy', data = all_results, color = 'green').set(title = 'Model Performance on Test Data - 10 Engineered Features', ylim = (0,1))








