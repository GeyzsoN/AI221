# %% [markdown]
# <main style="font-family: TeX Gyre Termes; font-size: 1.2rem">
# 
# ### MEX #2 - Geyzson Kristoffer
# SN:2023-21036
# 
# https://uvle.upd.edu.ph/mod/assign/view.php?id=535541
# 
# <hr>

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from icecream import ic

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.svm import SVC, SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression

from matplotlib.lines import Line2D


# %% [markdown]
# # Problem #1

# %%
penguin_data = pd.read_csv('penguins_size.csv')
penguin_data

# %%
ic(penguin_data.isna().sum())
ic(penguin_data.isnull().sum())

# %% [markdown]
# # Problem #1-a
# 

# %%
columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species']

penguin_data_clean = penguin_data[columns].dropna()

sns.pairplot(penguin_data_clean, hue='species', palette='Set2', diag_kind='kde', height=2)
plt.suptitle('Pair Plot of Penguin Features', y=1.02)
plt.show()


# %% [markdown]
# # Problem #1-b
# 

# %%
X = penguin_data_clean[['culmen_length_mm', 'flipper_length_mm']]
y = penguin_data_clean['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred, average='macro')
train_confusion = confusion_matrix(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='macro')
test_confusion = confusion_matrix(y_test, y_test_pred)

print(f'Train Accuracy: \t{train_accuracy:.5f}')
print(f'Train Macro Avg F1: \t{train_f1:.5f}')
cm_train = ConfusionMatrixDisplay(confusion_matrix = train_confusion)
cm_train.plot()
plt.title('Confusion Matrix for Training Data')
plt.show()


print(f'Test Accuracy: \t\t{test_accuracy:.5f}')
print(f'Test Macro Avg F1: \t{test_f1:.5f}')
cm_test = ConfusionMatrixDisplay(confusion_matrix = test_confusion)
cm_test.plot()
plt.title('Confusion Matrix for Testing Data')
plt.show()

# %% [markdown]
# # Problem #1-c
# 

# %%
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

display = DecisionBoundaryDisplay.from_estimator(model, X, alpha=0.8, eps=0.5)
plt.scatter(X_train['culmen_length_mm'], X_train['flipper_length_mm'], c=y_train_encoded, edgecolors="k", marker='o', label='Training Data')
plt.scatter(X_test['culmen_length_mm'], X_test['flipper_length_mm'], c=y_test_encoded, edgecolors="k", marker='X', label='Testing Data')
plt.title('Decision Boundary with Training and Testing Data')
plt.xlabel('Culmen Length (mm)')
plt.ylabel('Flipper Length (mm)')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=10, label='Training Data'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='k', markersize=10, label='Testing Data')
]
plt.legend(handles=legend_elements)

plt.show()

# %% [markdown]
# # Problem #1-d

# %%

parameters = {
    'svc__C': [0.01, 0.1, 1, 10, 100], 
    'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'svc__degree': [2, 3, 4, 5],
    'svc__decision_function_shape': ['ovr', 'ovo']
}

random_search = RandomizedSearchCV(model, parameters, n_iter=5, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred, average='macro')
train_confusion = confusion_matrix(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='macro')
test_confusion = confusion_matrix(y_test, y_test_pred)

print(f'Train Accuracy: \t{train_accuracy:.5f}')
print(f'Train Macro Avg F1: \t{train_f1:.5f}')
cm_train = ConfusionMatrixDisplay(confusion_matrix = train_confusion)
cm_train.plot()
plt.title('Confusion Matrix for Training Data')
plt.show()


print(f'Test Accuracy: \t\t{test_accuracy:.5f}')
print(f'Test Macro Avg F1: \t{test_f1:.5f}')
cm_test = ConfusionMatrixDisplay(confusion_matrix = test_confusion)
cm_test.plot()
plt.title('Confusion Matrix for Testing Data')
plt.show()

# %%
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

display = DecisionBoundaryDisplay.from_estimator(best_model, X, alpha=0.8, eps=0.5)
plt.scatter(X_train['culmen_length_mm'], X_train['flipper_length_mm'], c=y_train_encoded, edgecolors="k", marker='D', label='Training Data')
plt.scatter(X_test['culmen_length_mm'], X_test['flipper_length_mm'], c=y_test_encoded, edgecolors="k", marker='X', label='Testing Data')
plt.title('Decision Boundary with Training and Testing Data')
plt.xlabel('Culmen Length (mm)')
plt.ylabel('Flipper Length (mm)')

legend_elements = [
    Line2D([0], [0], marker='D', color='w', markerfacecolor='k', markersize=10, label='Training Data'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='k', markersize=10, label='Testing Data')
]
plt.legend(handles=legend_elements)

plt.show()

# %% [markdown]
# <hr>

# %% [markdown]
# # Problem #2

# %%
bike_data = pd.read_csv('SeoulBikeData.csv', encoding='latin1')
bike_data

# %% [markdown]
# # Problem #2-a

# %%
winter_data = bike_data[bike_data['Seasons'] == 'Winter']

plt.figure(figsize=(8, 8))

plt.subplot(3, 3, 1)
sns.boxplot(data=winter_data[['Hour']])
plt.title("Hour")

plt.subplot(3, 3, 2)
sns.boxplot(data=winter_data[['Temperature(°C)']])
plt.title("Temperature(°C)")

plt.subplot(3, 3, 3)
sns.boxplot(data=winter_data[['Humidity(%)']])
plt.title("Humidity(%)")

plt.subplot(3, 3, 4)
sns.boxplot(data=winter_data[['Wind speed (m/s)']])
plt.title("Wind speed (m/s)")

plt.subplot(3, 3, 5)
sns.boxplot(data=winter_data[['Visibility (10m)']])
plt.title("Visibility (10m)")

plt.subplot(3, 3, 6)
sns.boxplot(data=winter_data[['Dew point temperature(°C)']])
plt.title("Dew point temperature(°C)")

plt.subplot(3, 3, 7)
sns.boxplot(data=winter_data[['Solar Radiation (MJ/m2)']])
plt.title("Solar Radiation (MJ/m2)")

plt.subplot(3, 3, 8)
sns.boxplot(data=winter_data[['Rainfall(mm)']])
plt.title("Rainfall(mm)")

plt.subplot(3, 3, 9)
sns.boxplot(data=winter_data[['Snowfall (cm)']])
plt.title("Snowfall (cm)")

plt.tight_layout()
plt.show()


# %% [markdown]
# # Problem #2-b

# %%
X = winter_data[['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 
                 'Visibility (10m)', 'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 
                 'Rainfall(mm)', 'Snowfall (cm)']]
y = winter_data['Rented Bike Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_svr = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())  
])

model_svr.fit(X_train, y_train)

y_pred = model_svr.predict(X_test)

rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse_svr:.5f}')


# %% [markdown]
# # Problem #2-b fine tuning
# 

# %%
param_grid = {
    'svr__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'svr__C': [0.1, 1, 10, 100],
    'svr__epsilon': [0.01, 0.1, 1]
}

grid_search = GridSearchCV(model_svr, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_svr = grid_search.best_estimator_

y_pred_best_svr = best_svr.predict(X_test)

rmse_best_svr = np.sqrt(mean_squared_error(y_test, y_pred_best_svr))


print(f'Best parameters: {best_params}')
print(f'RMSE: {rmse_best_svr:.5f}')


# %% [markdown]
# # Problem #2-c
# 

# %%
model_krr = Pipeline([
    ('scaler', StandardScaler()),
    ('krr', KernelRidge(kernel='linear')) 
])

model_krr.fit(X_train, y_train)

y_pred_krr = model_krr.predict(X_test)

rmse_krr = np.sqrt(mean_squared_error(y_test, y_pred_krr))
print(f'RMSE original: {rmse_krr:.5f}')

param_grid_krr = {
    'krr__alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'krr__kernel': ['linear', 'poly', 'rbf'],
    'krr__degree': [2, 3, 4],
    'krr__coef0': [0, 1, 2],
    'krr__gamma': [0.01, 0.1, 1, 10]
}

randsearch_krr = RandomizedSearchCV(model_krr, param_grid_krr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
randsearch_krr.fit(X_train, y_train)

best_params = randsearch_krr.best_params_
best_krr = randsearch_krr.best_estimator_

y_pred_krr = best_krr.predict(X_test)
rmse_krr = np.sqrt(mean_squared_error(y_test, y_pred_krr))
print(f'Best parameters: {best_params}')
print(f'RMSE fine tuned: {rmse_krr:.5f}')


# %% [markdown]
# # Problem #2-d

# %%
model_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(f'RMSE original: {rmse_lr:.5f}')

param_grid_lr = {
    'lr__fit_intercept': [True, False]
}

grid_lr = GridSearchCV(model_lr, param_grid_lr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_lr.fit(X_train, y_train)

best_params = grid_lr.best_params_
best_lr = grid_lr.best_estimator_

y_pred_lr = best_lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(f'Best parameters: {best_params}')
print(f'RMSE fine tuned: {rmse_lr:.5f}')


