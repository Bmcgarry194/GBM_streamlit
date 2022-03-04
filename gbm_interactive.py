import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

import matplotlib.pyplot as plt

import streamlit as st

st.title("Gradient Boosted Model Explorer")



st.markdown(r'''
## Generate and Visualize Synthetic Data

We are going to create synthetic data so know exactly how the data is generated and the true underlying pattern is available for us to examine.

For the sake of visualization the target variable is only dependant on a single independant variable `x` and takes on the mathematical form:

$$ y = \beta_1 * \sin{x} + \beta_2 * x + b$$
''')
st.sidebar.header('''Data Generator''')

n_obs = st.sidebar.number_input("Number of Observations", value=30)
beta_1 = st.sidebar.number_input("Beta 1 Coefficient Value", value=4)
beta_2 = st.sidebar.number_input("Beta 2 Coefficient Value", value=2)
intercept = st.sidebar.number_input("Intercept Value", value=1)
noisiness = st.sidebar.number_input("Noisiness: The extent to which the observations deviate from the ground truth pattern", value=2)

rng = np.random.default_rng(2)

x = rng.uniform(0, 10, size=n_obs)
noise = rng.normal(0, noisiness, size=n_obs)

signal = beta_1 * np.sin(x) + beta_2 * x + intercept

y = signal + noise

X_train, X_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y)



fig, ax = plt.subplots(figsize=(15, 10))
pred_range = np.linspace(0, 10, num=1000).reshape(-1, 1)

ax.scatter(X_train, y_train, label='Train')
ax.scatter(X_test, y_test, label='Test')
ax.plot(pred_range, beta_1 * np.sin(pred_range) + beta_2 * pred_range + intercept, label='Ground Truth', color='black', linestyle='--', alpha=.5)

ax.legend()
st.pyplot(fig)

st.markdown('''
You can customize the generated dataset using the Data Generator options on the sidebar.

Now we will show how a Gradient Boosted Machine can approximate this pattern using layers of decision trees.
''')

st.markdown('''

## Train a Gradient Boosting Regression

To understand how gradient boosting learns a pattern lets start with a very basic version where each decision tree is only allowed to make a **single split(max_depth=1)**, the **learning rate is 1** and there are only **5 decsion trees** used total.
''')

st.sidebar.header('''GBM Hyperparameters''')

num_trees = st.sidebar.number_input("Number of Trees", 1, value=5)
max_depth = st.sidebar.number_input("Maximum Depth of Individual Trees", 1, value=1)
lr = st.sidebar.number_input("Learning Rate", 0.0, 1.0, value=1.0)

gb = GradientBoostingRegressor(n_estimators=num_trees, max_depth=max_depth, learning_rate=lr)

gb.fit(X_train, y_train)

pred_range = np.linspace(0, 10, num=500).reshape(-1, 1)
preds = gb.predict(pred_range)

col1, col2 = st.columns(2)
col1.metric('Training R^2', f'{gb.score(X_train, y_train):.3f}')
col2.metric('Test R^2', f'{gb.score(X_test, y_test):.3f}')

fig, ax = plt.subplots(figsize=(15, 9))

ax.plot(pred_range, beta_1 * np.sin(pred_range) + beta_2 * pred_range + intercept, label='Ground Truth', color='black', linestyle='--', alpha=.5)
ax.scatter(X_train, y_train, label='Train')
ax.scatter(X_test, y_test, label='Test')
    
ax.plot(pred_range, preds, label='Model Predictions', color='darkgreen', alpha=.75)
    
ax.legend()
st.pyplot(fig)

st.markdown('''
Take some time to mess around with the hyperparameters of the GBM in the sidebar. The model is only trained on the blue training points. Notice how it affects both the training and test R^2 values.

Can you find the best combination of Number of Trees, Max Depth and Learning Rate to maximize the **Training R^2**?

What about the **Testing R^2**?

Think about how each of these hyperparaters affects the complexity of the model. Does increasing the value of the Number of Trees increase the complexity of the model or decrease it? What about the other two hyperparameters?
''')

st.markdown('''

## Visualize the decision trees at each step

In the first image notice that we have already centered our data by subtracting off the mean from each data point. 

The first single decision tree must decide where to split the dataset so that the two resulting regions means will minimize the error. In this case the split that minimizes the error is located around 5.5. This indicates that if an observation has a value <5.5 the model will predict -5 and when it is >5.5 the prediction is around 8.

The data points in the next tree down are the residuals from the previous tree.

''')


C = y_train.mean()
residuals = y_train - C


fig, axes = plt.subplots(num_trees, figsize=(10, 4*num_trees), sharey=True, sharex=True)

for i, (ax, tree) in enumerate(zip(axes, gb.estimators_)):
    
    tree_preds = tree[0].predict(pred_range)
    
    ax.scatter(X_train, residuals, color='orange')
    
    ax.vlines(x=X_train[1], ymin=residuals[1], ymax=tree[0].predict(X_train[1].reshape(-1, 1)), color='black', linestyle='--')
    
    residuals -= tree[0].predict(X_train)
    
    ax.plot(pred_range, tree_preds)
    
    ax.set_title(f'Tree {i}')
st.pyplot(fig)

st.markdown('''
## Cumulative Predictions of the Trees

As we add in the contributions to later trees we can see the model getting more and more complex and getting a better fit to the training data.
''')
    
fig, axes = plt.subplots(num_trees, figsize=(15, 5*num_trees))

tree_num = 1

for ax, preds, y_pred_train, y_pred_test in zip(axes, gb.staged_predict(pred_range), gb.staged_predict(X_train), gb.staged_predict(X_test)):
    
    ax.set_title(f'{tree_num} Tree(s)  Training R2: {r2_score(y_train, y_pred_train):.2f} Testing R2: {r2_score(y_test, y_pred_test):.2f}')
    tree_num += 1
    
    ax.plot(pred_range, beta_1 * np.sin(pred_range) + beta_2 * pred_range + intercept, label='Ground Truth', color='black', linestyle='--', alpha=.5)
    ax.scatter(X_train, y_train, label='Train')
    ax.scatter(X_test, y_test, label='Test')
    
    ax.plot(pred_range, preds, label='Model Predictions', color='darkgreen', alpha=.75)
    
    ax.legend()
st.pyplot(fig)


st.markdown('''
## Overlapping Trees

This graph shows all of the cumulative trees at each step of the boosting process. Its easier to see that as we add trees to our GBM they allow our predictions to better fit to our training data
''')
fig, ax = plt.subplots(figsize=(12, 8))

for i, (preds, y_pred_train, y_pred_test) in enumerate(zip(gb.staged_predict(pred_range), gb.staged_predict(X_train), gb.staged_predict(X_test))):
    if i == num_trees - 1:
        ax.plot(pred_range, preds, label=f'Model Predictions with {i+1} Trees', alpha=1)
    else:
        ax.plot(pred_range, preds, label=f'Model Predictions with {i+1} Trees', alpha=.3)

ax.plot(pred_range, beta_1 * np.sin(pred_range) + beta_2 * pred_range + intercept, label='Ground Truth', color='black', linestyle='--', alpha=.5)
        
ax.scatter(X_train, y_train, label='Train')
ax.scatter(X_test, y_test, label='Test')

ax.legend()
st.pyplot(fig)