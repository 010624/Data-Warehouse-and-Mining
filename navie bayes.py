import numpy as np
import pandas as pd

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]
            
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        
        for c in self.classes:
            prior = np.log(self.priors[c])
            posterior = np.sum(np.log(self._pdf(c, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

# Create our own dataset
data = {
    'Milk': [1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
    'Bread': [1, 1, 0, 1, 1, 0, 0, 0, 1, 1],
    'Butter': [0, 0, 1, 0, 1, 1, 1, 0, 0, 1],
    'Jam': [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    'Eggs': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Generate arbitrary labels (for demonstration purposes)
df['target'] = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier and fit the model
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb.predict(X_test)

# Evaluate the model
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"Accuracy: {accuracy:.2f}")

# Displaying predictions
print("Predictions:")
for i, pred in enumerate(y_pred):
    print(f"Sample {i+1}: Predicted Class = {pred}, Actual Class = {y_test[i]}")


# Aim : Implementation of Naive Bayes algorithm 
# Theory : 
# • Naive Bayes is a machine learning algorithm for classification problems. It is based on Bayes’ probability 
# theorem. It is primarily used for text classification which involves high dimensional training data sets. 
# • The Naive Bayes algorithm is called “naive” because it makes the assumption that the occurrence of a certain 
# feature is independent of the occurrence of other features. 
# Implementation of Naive Bayes Algorithms : 
# We are using the Social network ad dataset. The dataset contains the details of users in a social networking 
# site to find whether a user buys a product by clicking the ad on the site based on their salary, age, and gender. 
# Steps involved for implementation : 
# 1. Importing essential libraries required. 
# 2. Importing of social network ad dataset. 
# 3. Since our dataset containing character variables we have to encode it using LabelEncoder. 4. performing a train test split on our dataset. 
# 5. Next, we are doing feature scaling to the training and test set of independent variables. 6. Training the Naive Bayes model on the training set. 
# 7. Let’s predict the test results. 
# 8. Making the Confusion Matrix 

#Conclusion : we have dealt with the Naive Bayes algorithm, we have covered most concepts of itin machine learning.


