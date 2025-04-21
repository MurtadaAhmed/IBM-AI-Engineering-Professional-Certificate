# Introduction to Logistic Regression

This guide provides an overview of logistic regression, a statistical modeling technique used in machine learning to predict probabilities and classify observations into one of two classes (e.g., true/false, 0/1). Below, the key concepts, applications, and mechanics of logistic regression are organized and explained.

## What is Logistic Regression?

- **Definition**: Logistic regression is a statistical modeling technique that predicts the probability of an observation belonging to one of two classes (binary classification).
- **Machine Learning Context**: In machine learning, logistic regression refers to a binary classifier derived from statistical logistic regression.
- **Functionality**:
  - **Probability Predictor**: Estimates the probability of an event (e.g., probability of a customer churning).
  - **Binary Classifier**: Assigns observations to one of two classes based on a threshold probability (e.g., 0.5).
    - If the predicted probability (p-hat) > threshold, assign to class 1.
    - If p-hat ≤ threshold, assign to class 0.

## When to Use Logistic Regression

Logistic regression is a suitable choice in the following scenarios:

1. **Binary Target**:
   - The target variable is binary (e.g., 0 or 1, yes or no).
   - Example: Predicting whether a customer will churn (yes/no).

2. **Need for Probabilistic Outcomes**:
   - When the goal is to estimate the probability of an outcome.
   - Example: Probability of a customer purchasing a product.

3. **Linearly Separable Data**:
   - The decision boundary can be represented as a line, plane, or hyperplane.
   - Mathematical form: θ₀ + θ₁x₁ + θ₂x₂ > 0, where θ₀ is the intercept, and θ₁, θ₂ are coefficients.

4. **Feature Impact Analysis**:
   - Allows understanding the influence of independent features based on their model coefficients (weights).
   - Useful for feature selection by identifying the most impactful features.

## Applications of Logistic Regression

Logistic regression is widely used to predict probabilities and classify outcomes in various domains, including:

- **Healthcare**:
  - Predicting the probability of a heart attack based on age, sex, and body mass index.
  - Estimating the likelihood of a disease (e.g., diabetes) based on patient characteristics like weight, height, blood pressure, and test results.
- **Business**:
  - Predicting the likelihood of a customer halting a subscription (churn).
  - Estimating the probability of a homeowner defaulting on a mortgage.
- **Engineering**:
  - Assessing the probability of failure in a process, system, or product.

## Example: Predicting Customer Churn

Consider a telecommunication dataset with historical customer data, where the goal is to predict which customers might churn (leave) next month. The dataset includes:

- **Features**: Services subscribed, account information, demographic details (e.g., gender, age).
- **Target Variable**: Churn (binary: 1 for churn, 0 for no churn).

### Modeling Approach
- Logistic regression uses one or more features (e.g., age, income) to predict:
  - **p-hat**: The probability that a customer will churn (y = 1) given the input features (x).
  - **y-hat**: The predicted class (churn or no churn) based on a threshold (e.g., p-hat > 0.5 → churn).

## Logistic Regression vs. Linear Regression

### Linear Regression Limitations
- **Problem**: Linear regression fits a line (y-hat = θ₀ + θ₁x₁) that predicts continuous values, which can extend indefinitely (e.g., beyond 0 or 1).
- **Issue for Binary Classification**: Predicted values are not constrained to [0, 1], making it unsuitable for probability prediction or binary classification.
- **Attempted Fix**:
  - Apply a threshold (e.g., y-hat < 0.5 → class 0, y-hat ≥ 0.5 → class 1).
  - This creates a step function, but it lacks smoothness and doesn’t provide probability estimates.
  - Example Issue: A 20-year-old and a 100-year-old customer might both be classified as churn (class 1) with no differentiation in probability.

### Logistic Regression Solution
- **Sigmoid Function**: Logistic regression uses the sigmoid (logit) function to map predictions to the range [0, 1].
  - Formula: σ(x) = 1 / (1 + e⁻ˣ)
  - Properties:
    - At x = 0, σ(x) = 0.5.
    - As x increases, σ(x) approaches 1.
    - As x decreases, σ(x) approaches 0.
  - The sigmoid function compresses any continuous input into a probability between 0 and 1.
- **Model Output**:
  - The model predicts p-hat = σ(y-hat), representing the probability that y = 1 given x.
  - A decision boundary (e.g., p-hat = 0.5) assigns classes:
    - p-hat > 0.5 → class 1.
    - p-hat ≤ 0.5 → class 0.

## Mechanics of Logistic Regression

### Model Representation
- **americaInput Features (x)**: Variables like age, income, etc.
- **Target Variable (y)**: Binary outcome (e.g., churn = 1, no churn = 0).
- **Model Parameters**:
  - θ₀: Intercept.
  - θ₁, θ₂, ...: Weights (coefficients) for each feature.
- **Prediction**:
  - Linear combination: y-hat = θ₀ + θ₁x₁ + θ₂x₂ + ...
  - Probability: p-hat = σ(y-hat) = 1 / (1 + e⁻(θ₀ + θ₁x₁ + θ₂x₂ + ...)).
- **Classification**: Assign class based on a threshold (e.g., 0.5).

### Decision Boundary
- The decision boundary is where p-hat = 0.5, corresponding to θ₀ + θ₁x₁ + θ₂x₂ = 0.
- This boundary is a line (in 2D), plane (in 3D), or hyperplane (in higher dimensions) for linearly separable data.

### Probability Interpretation
- **Churn Probability**: p(y = 1 | x) = p-hat (e.g., 0.8 for churn).
- **Non-Churn Probability**: p(y = 0 | x) = 1 - p-hat (e.g., 1 - 0.8 = 0.2).
- The sum of probabilities for both classes equals 1.

## Visualizing Logistic Regression

- **Scatterplot**:
  - Plot data points with features (e.g., age on x-axis) and binary target (e.g., churn: 0 in red, 1 in blue).
- **Linear Regression**:
  - Fits a straight line (y-hat = θ₀ + θ₁x₁), but predictions exceed [0, 1].
- **Logistic Regression**:
  - Applies the sigmoid function to produce a smooth curve that maps predictions to [0, 1].
  - The decision boundary (p-hat = 0.5) separates the classes.

## Key Takeaways

- **Logistic Regression Overview**:
  - A binary classifier and probability predictor based on statistical logistic regression.
  - Uses the sigmoid function to constrain predictions to [0, 1].
- **When to Use**:
  - Binary target variables.
  - Need for probabilistic outcomes.
  - Linearly separable data.
  - Feature impact analysis.
- **Applications**:
  - Predicting churn, disease likelihood, mortgage defaults, system failures, etc.
- **Advantages**:
  - Provides interpretable probabilities.
  - Allows feature selection based on coefficient magnitudes.
  - Handles binary classification effectively with a smooth decision boundary.

By understanding logistic regression, you can build models to predict binary outcomes and estimate probabilities, making it a powerful tool in machine learning and data analysis.