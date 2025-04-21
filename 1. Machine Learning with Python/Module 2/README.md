Introduction to regression:
- define regression: supervised learning model, find relationship between target variable and features
example: predict co2 emission of car.
- simple vs multiple regression

Simple Regression: single independent variable estimates a dependent variable.
two types: Linear or nonlinear (example determine co2 emission using engine size)

Mutiple Regression: more than one independend variable (example determine co2 emission using engine size and number of cylinders)
two types: Linear or nonlinear

- applications of regression:
1. sales forcasting
2. price estimation
3. predictive maintenance
4. employment income
5. estimate rainfall
6. wildfire propabaility
7. spread of infectious disease
8. risk of chronic disease


- Regression algorithms:
1. Linear and plynomial
2. Random forest
3. Extrene gradient boosting (XGBoost)
4. K-nearest neighbors(KNN)
5. Support vector machines (SVM)
6. Neural network

------------------------
Simple Linear Regression:
- Describe:
y = theta0 + theta1 * X1
y: response variable
X1: single predictor

residual error: is the vertical distance between the data point to the fitted regression line.
the average of all residual errors measure how poorly the regression line fits the data.
can be shown using mean square error (MSE)

Linear regression aims to find the line for minimizing all these residual errors.>> OLS regression (ordinary least squares regression)
theta0 >> bias coefficient
theta1 >> coeffient for co2 emission column

OLS is easy to understand and interpret. Doesn't require any tunning, as the solution is just calculation. However, outliers can greatly reduce its accuracy.

------------------------
Multiple Linear Regression:
- Describe
extenstion of simple linear regression
uses two or more independent variable to estimate a dependant variable
* better than simple linear regression
* too many variables can cause overfitting
* convert categorical independent variables into numerical variables


- multiple linear regression vs simple linear regression
- pitfalls of multiple linear regression

applications:
- education >> check exam performance based on revision time, test anxiety, attendance, gender
- what-if scenarios

select variables that are:
- most understood
- controllable
- most correlated with target


with two features >> solution describes a plane
beyound two dimentions >> solution describes a hyperplane

Least-squares linear regression

estimating parameters:
- Ordiniray least squares >> estimate coefficients by minimising MSE
- Gradinet descent (optamization with random values)


You might choose Lasso regression when you suspect that many of your independent variables are not actually useful for predicting the dependent variable. Lasso can help by shrinking some coefficients to zero, effectively removing those variables from the model. This can lead to a simpler, more interpretable model.

On the other hand, you might choose Ridge regression when you believe that all the independent variables are relevant, but you want to prevent overfitting. Ridge regression keeps all variables in the model but reduces their influence by penalizing large coefficients.

While MSE is a widely used metric, it can be sensitive to outliers. In cases where your data contains significant outliers, MSE might give a misleading impression of model performance because the squared errors of these outliers can disproportionately affect the overall score.

In such scenarios, you might consider using Mean Absolute Error (MAE) instead. MAE measures the average absolute differences between predicted and actual values, which makes it less sensitive to outliers compared to MSE.

------------
Polynomial and Nonlinear Regression:

Nonlinear machine learning models:
Regression trees
Random forests
Neural networks
Support Vector Machines
Gradient Boosting Machines
K-Nearest Neighbors


Key Concepts:

Polynomial Regression:
    Definition: A type of regression that uses linear regression to fit data to polynomial expressions of the features.
    Modeling: The relationship between the independent variable ( x ) and the dependent variable ( y ) is modeled as an nth degree polynomial: [ y = \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3 + \ldots + \theta_n x^n ]
    Linearization: By introducing new variables (e.g., ( x_1 = x, x_2 = x^2, x_3 = x^3 )), the polynomial regression can be expressed as a linear combination, allowing the use of ordinary linear regression techniques.

Nonlinear Regression:
    Definition: A statistical method for modeling relationships where the equation is nonlinear. This includes polynomial, exponential, logarithmic, and other functions.
    Use Cases: Nonlinear regression is particularly useful when data exhibits complex relationships that cannot be adequately modeled with a straight line.
Overfitting:
    Concern: A polynomial regression model can perfectly fit the training data by memorizing every point, including noise, leading to overfitting. This means the model may not generalize well to new data.
    Solution: It’s crucial to choose a model that captures the underlying trend without fitting every detail.

Applications of Nonlinear Regression:

Exponential Growth: For example, modeling how investments grow with compound interest.
Logarithmic Relationships: Such as the law of diminishing returns, where productivity gains decrease as more resources are invested.
Periodic Functions: For instance, modeling seasonal variations in data like rainfall or temperature.

Practical Steps for Model Selection:

Visual Analysis: Use scatter plots to analyze the relationship between the target variable and input variables. Look for patterns that suggest linear, exponential, logarithmic, or sinusoidal relationships.
Model Generation: Create different models based on the identified patterns and analyze their performance.
Error Analysis: Plot predictions against actual values to visually interpret model errors.

Finding Optimal Models:

Optimization Techniques: If you have a mathematical expression for your model, techniques like gradient descent can help find the best parameters.
Machine Learning Models: If unsure about the specific regression model, consider using machine learning models such as:
    Regression Trees
    Random Forests
    Neural Networks
    Support Vector Machines
    Gradient Boosting Machines
    K-Nearest Neighbors

Summary:

Nonlinear regression, including polynomial regression, is essential for modeling complex relationships in data. It allows for flexibility in capturing trends while being cautious of overfitting. Understanding the nature of your data and selecting the appropriate model is key to effective analysis.



------------------------











