#in terminal copy the this line (pip install mlxtend)
#************************************************************pip install mlxtend********************************************************************************
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transaction-like data for demonstration
data = {
    'Item 1': [1, 0, 0, 1, 1],
    'Item 2': [0, 1, 1, 0, 1],
    'Item 3': [1, 1, 0, 1, 0],
    'Item 4': [0, 1, 1, 0, 1],
    'Item 5': [1, 1, 0, 1, 1],
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Step 1: Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# Step 2: Generate association rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display the frequent itemsets and the rules
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)
