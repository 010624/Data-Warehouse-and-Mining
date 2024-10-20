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

