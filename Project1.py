# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn import datasets, metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, StackingClassifier


""" Data Processing """
df = pd.read_csv("data/Project_1_Data.csv")

""" Data Visualization """
# Basic histograms to analyze raw data.
fig1, axs1 = plt.subplots(nrows = 1, ncols = 4, figsize = (11, 4))

axs1[0].hist(df["X"], bins = 10, color = 'Blue', edgecolor = 'black')
axs1[0].set_title('Frequency of X')
 
axs1[1].hist(df["Y"], bins = 10, color = 'Green', edgecolor = 'black')
axs1[1].set_title('Frequency of Y')

axs1[2].hist(df["Z"], bins = 10, color = 'Yellow', edgecolor= 'black')
axs1[2].set_title('Frequency of Z')
 
axs1[3].hist(df["Step"], bins = 13, color = 'Red', edgecolor='black')
axs1[3].set_title('Step Size')

for ax in axs1:
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Histograms don't tell much apart from frequency of values. 3D plot is better for this visualization.
fig2 = plt.figure()
ax = fig2.add_subplot(projection = "3d")

step_Value = df["Step"].unique()    # To get unique step values.
for i, val in enumerate(step_Value):
    temporary_df = df[
        df["Step"] == val
        ]
    ax.scatter(
            temporary_df["X"], temporary_df["Y"], temporary_df["Z"], marker = ".", label = f"Step: {val}")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

fig2.subplots_adjust(bottom = -5)
fig2.suptitle("\n 3D Visualization of Each Coordinate \n and Its Corresponding Step")
fig2.legend(loc = "lower center", title = "Steps", ncols = 4, bbox_to_anchor = (0.53, -0.32))

plt.tight_layout()
plt.show()


""" Stratified Sampling """
# Establish Independent and Dependent Variables.
# X = ['X',
#      'Y',
#      'Z']
# Y = ['Step']

my_splitter = StratifiedShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 42)

for train_index, test_index in my_splitter.split(df, df["Step"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)

# For Loop to See Train Values Distribution Across All Steps.
step_uniqValue = strat_df_train["Step"].unique()
for i, val in enumerate(step_uniqValue):
    step_train_distrib = strat_df_train['Step'].value_counts()
print(f"\nDistribution of Train Data in Each Step:\n {step_train_distrib}")

# For Loop to See Test Values Distribution Across All Steps.
step_uniqValue2 = strat_df_test["Step"].unique()
for i, val in enumerate(step_uniqValue2):
    step_test_distrib = strat_df_test['Step'].value_counts()
print(f"\nDistribution of Test Data in Each Step:\n {step_test_distrib}")

X_train = strat_df_train.drop("Step", axis = 1)
Y_train = strat_df_train["Step"]
X_test = strat_df_test.drop("Step", axis = 1)
Y_test = strat_df_test["Step"]

""" Correlation Matrix """
# Correlation Matrix Needed to Determine Collinearity of the Variables.
corr_matrix = strat_df_train.corr()
corr_matrix2 = X_train.corr()

fig4, axs2 = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 4))
sns.heatmap(np.abs(corr_matrix), ax = axs2[0])
sns.heatmap(np.abs(corr_matrix2), ax = axs2[1])

axs2[0].set_title('With Step')
axs2[1].set_title('Without Step')

# Correlation Coeffiecients
corr1 = Y_train.corr(X_train['X'])
print(f"X and Step Correlation Coefficient: {corr1}")
corr2 = Y_train.corr(X_train['Y'])
print(f"Y and Step Correlation Coefficient: {corr2}")
corr3 = Y_train.corr(X_train['Z'])
print(f"Z and Step Correlation Coefficient: {corr3}")


""" Model Training """
# Logistic Regression
logistic_reg = LogisticRegression(solver = 'lbfgs', max_iter = 1000, random_state = 42)
param_grid_lr = {}
grid_search_lr = GridSearchCV(logistic_reg, param_grid_lr, cv = 5, scoring = 'neg_log_loss', n_jobs = -1)
grid_search_lr.fit(X_train, Y_train)
best_model_lr = grid_search_lr.best_estimator_
print("\nBest Logistic Regression Model:", best_model_lr)

# Training and testing error for Logistic Regression
Y_train_pred_lr = best_model_lr.predict_proba(X_train)
Y_test_pred_lr = best_model_lr.predict_proba(X_test)
CE_train_lr = log_loss(Y_train, Y_train_pred_lr)
CE_test_lr = log_loss(Y_test, Y_test_pred_lr)
print(f"Logistic Regression - CE Loss (Train): {CE_train_lr}, CE Loss (Test): {CE_test_lr}")


# Decision Tree (Randomized Search Cross-Validation Model)
decision_tree = DecisionTreeClassifier(random_state = 42)
param_grid_dt = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [100, 120, 140],
    'min_samples_leaf': [50, 100, 150],
    'max_features': ['sqrt', 'log2']
}
grid_search_dt = RandomizedSearchCV(decision_tree, param_grid_dt, cv = 5, scoring = 'f1_micro', n_jobs = -1)
grid_search_dt.fit(X_train, Y_train)
best_model_dt = grid_search_dt.best_estimator_
print("\nBest Decision Tree Model:", best_model_dt)

# Training and testing error for Decision Tree
Y_train_pred_dt = best_model_dt.predict(X_train)
Y_test_pred_dt = best_model_dt.predict(X_test)
F1_train_dt = f1_score(Y_train, Y_train_pred_dt, average = 'micro')
F1_test_dt = f1_score(Y_test, Y_test_pred_dt, average = 'micro')
print(f"Decision Tree - F1 Score (Train): {F1_train_dt}, F1 Score (Test): {F1_test_dt}")

# Random Forest
random_forest = RandomForestClassifier(random_state = 42)
param_grid_rf = {
    'n_estimators': [10, 30, 50, 100, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}   
grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv = 5, scoring = 'f1_micro', n_jobs = -1)
grid_search_rf.fit(X_train, Y_train)
best_model_rf = grid_search_rf.best_estimator_
print("\nBest Random Forest Model:", best_model_rf)

# Training and testing error for Random Forest
Y_train_pred_rf = best_model_rf.predict(X_train)
Y_test_pred_rf = best_model_rf.predict(X_test)
F1_train_rf = f1_score(Y_train, Y_train_pred_rf, average = 'micro')
F1_test_rf = f1_score(Y_test, Y_test_pred_rf, average = 'micro')
print(f"Random Forest - F1 Score (Train): {F1_train_rf}, F1 Score (Test): {F1_test_rf}")

# Support Vector Machine (SVM)
svr = SVC(random_state = 42, class_weight = 'balanced')
param_grid_svr = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4, 5, 6],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}
grid_search_svr = GridSearchCV(svr, param_grid_svr, cv = 5, scoring = 'f1_micro', n_jobs = -1)
grid_search_svr.fit(X_train, Y_train)
best_model_svr = grid_search_svr.best_estimator_
print("\nBest SVM Model:", best_model_svr)

# Training and testing error for SVM
Y_train_pred_svr = best_model_svr.predict(X_train)
Y_test_pred_svr = best_model_svr.predict(X_test)
F1_train_svr = f1_score(Y_train, Y_train_pred_svr, average = 'micro')
F1_test_svr = f1_score(Y_test, Y_test_pred_svr, average = 'micro')
print(f"SVC - F1 Score (Train): {F1_train_svr}, F1 Score (Test): {F1_test_svr}")


""" Model Performance Analysis """
# Logistic Regression
Y_test_pred_rounded_lr = np.argmax(Y_test_pred_lr, axis=1)
acc_score_lr = accuracy_score(Y_test, Y_test_pred_rounded_lr)
prec_score_lr = precision_score(Y_test, Y_test_pred_rounded_lr, average = 'micro')
f1_score_lr = f1_score(Y_test, Y_test_pred_rounded_lr, average = 'micro')
print(f"\n Logistic Regression - Accuracy: {acc_score_lr}, Precision: {prec_score_lr}, F1 Score: {f1_score_lr}")

# Decision Tree
acc_score_dt = accuracy_score(Y_test, Y_test_pred_dt)
prec_score_dt = precision_score(Y_test, Y_test_pred_dt, average = 'micro')
f1_score_dt = f1_score(Y_test, Y_test_pred_dt, average = 'micro')
print(f"\n Decision Tree - Accuracy: {acc_score_dt}, Precision: {prec_score_dt}, F1 Score: {f1_score_dt}")

# Random Forest
acc_score_rf = accuracy_score(Y_test, Y_test_pred_rf)
prec_score_rf = precision_score(Y_test, Y_test_pred_rf, average = 'micro')
f1_score_rf = f1_score(Y_test, Y_test_pred_rf, average = 'micro')
print(f"\n Random Forest - Accuracy: {acc_score_rf}, Precision: {prec_score_rf}, F1 Score: {f1_score_rf}")

# Support Vector Machine
acc_score_svr = accuracy_score(Y_test, Y_test_pred_svr)
prec_score_svr = precision_score(Y_test, Y_test_pred_svr, average = 'micro')
f1_score_svr = f1_score(Y_test, Y_test_pred_svr, average = 'micro')
print(f"\n SVC - Accuracy: {acc_score_svr}, Precision: {prec_score_svr}, F1 Score: {f1_score_svr}")


""" Confusion Matrix """
# Random Forest Model is the model.
cnf_matrix = confusion_matrix(Y_test, Y_test_pred_rf)

class_names=[1,2,3,4,5,6,7,8,9,10,11,12,13]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(np.abs(cnf_matrix), annot = True, cmap = "YlGnBu" , fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix of Random Forest Classifier', y = 1.1)
plt.ylabel('Actual Steps')
plt.xlabel('Predicted Steps')


""" Stacked Model Performance """
estimators = []
estimators.append(('Decision Tree Classifier', best_model_dt))
estimators.append(('Random Forest Classifier', best_model_rf))

stacked_classifier = StackingClassifier(estimators = estimators, final_estimator = logistic_reg, cv = "prefit")
stacked_classifier.fit(X_train, Y_train)

Y_test_pred_stk = stacked_classifier.predict(X_test)
acc_score_stk = accuracy_score(Y_test, Y_test_pred_stk)
prec_score_stk = precision_score(Y_test, Y_test_pred_stk, average = 'micro')
f1_score_stk = f1_score(Y_test, Y_test_pred_stk, average = 'micro')
print(f"\n Stacked Classifier - Accuracy: {acc_score_stk}, Precision: {prec_score_stk}, F1 Score: {f1_score_stk}")

""" Confusion Matrix for Stacked Classifier """
cnf_matrix_stk = confusion_matrix(Y_test, Y_test_pred_stk)

class_names=[1,2,3,4,5,6,7,8,9,10,11,12,13]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(np.abs(cnf_matrix_stk), annot = True, cmap = "YlGnBu" , fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix of Stacked Classifier', y = 1.1)
plt.ylabel('Actual Steps')
plt.xlabel('Predicted Steps')


""" Model Evaluation """
# Saving the Chosen Model in a Joblib Format.
final_model = joblib.dump(best_model_rf,'random_forest_best_model.joblib')

# Another Excel CSV File was Created for Evaluation.
df = pd.read_csv("data/Project_1_Evaluation_Data.csv")

# Loading the Saved Joblib File.
final_model = joblib.load('random_forest_best_model.joblib')

X_test_eval_columns = ['X',
              'Y',
              'Z']
X_test_eval = df[X_test_eval_columns]

# Make Predictions Based on the Chosen Model.
predictions = final_model.predict(X_test_eval)

# Show Prediction Results. (Predictions: 5, 8, 13, 6, 4)
print("\n The Predictions Are:", predictions)


