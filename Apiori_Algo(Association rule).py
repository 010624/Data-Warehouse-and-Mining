import pandas as pd
from itertools import combinations

def load_data():
    # Sample data: Each row represents a transaction
    data = {
        'Milk': [1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
        'Bread': [1, 1, 0, 1, 1, 0, 0, 0, 1, 1],
        'Butter': [0, 0, 1, 0, 1, 1, 1, 0, 0, 1],
        'Jam': [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        'Eggs': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    }
    return pd.DataFrame(data)

def create_candidates(df, k):
    Ck = {}
    for _, transaction in df.iterrows():
        items = list(transaction[transaction == 1].index)
        for combo in combinations(items, k):
            combo = tuple(sorted(combo))
            if combo in Ck:
                Ck[combo] += 1
            else:
                Ck[combo] = 1
    return Ck

def filter_candidates(Ck, min_support, total_transactions):
    Lk = {}
    for key in Ck:
        support = Ck[key] / total_transactions
        if support >= min_support:
            Lk[key] = support
    return Lk

def apriori(df, min_support=0.5):
    total_transactions = len(df)
    L = []
    C1 = create_candidates(df, 1)
    L1 = filter_candidates(C1, min_support, total_transactions)
    L.append(L1)
    
    k = 2
    while True:
        Ck = create_candidates(df, k)
        Lk = filter_candidates(Ck, min_support, total_transactions)
        if not Lk:
            break
        L.append(Lk)
        k += 1
        
    return L

def generate_rules(L, min_confidence=0.7):
    rules = []
    for Lk in L[1:]:
        for itemset in Lk:
            subsets = [set(x) for x in combinations(itemset, len(itemset)-1)]
            for subset in subsets:
                remain = set(itemset) - subset
                if not remain:
                    continue
                subset = tuple(subset)
                confidence = Lk[itemset] / L[len(subset)-1][subset]
                if confidence >= min_confidence:
                    rules.append((subset, tuple(remain), confidence))
    return rules

# Load sample data
df = load_data()

# Run Apriori algorithm
min_support = 0.3
L = apriori(df, min_support)

# Generate rules from frequent itemsets
min_confidence = 0.7
rules = generate_rules(L, min_confidence)

# Display the results
print("Frequent Itemsets:")
for i, Lk in enumerate(L, start=1):
    print(f"L{i}: {Lk}")

print("\nAssociation Rules:")
for rule in rules:
    print(f"Rule: {rule[0]} -> {rule[1]}, confidence: {rule[2]:.2f}")

"""
Association Rule Mining is a data mining technique used to discover relationships or patterns among a large set of data items. 
It is commonly applied in market basket analysis, where the goal is to find associations between items frequently bought 
together by customers. The algorithm identifies rules that highlight how the occurrence of one item or set of items in a 
transaction affects the occurrence of another.

Apriori Algorithm
The Apriori algorithm is one of the most commonly used methods for association rule mining. It operates in two main phases:

Frequent Itemset Generation: Identifies all itemsets that meet a minimum support threshold.
Rule Generation: Generates association rules from the frequent itemsets, using a minimum confidence threshold.
Steps of the Apriori Algorithm
Initialize: Start by finding all itemsets of size 1 that meet the minimum support threshold.

Generate larger itemsets: Combine the itemsets of size k to generate itemsets of size 
k+1. Filter those that do not meet the support threshold.

Repeat: Continue generating larger itemsets until no more itemsets meet the minimum support.

Generate rules: For each frequent itemset, generate all possible rules that meet the minimum confidence threshold.


Advantages
Simplicity: The algorithm is easy to implement and understand.
Effectiveness: Useful in market basket analysis for understanding customer behavior.


Limitations
Scalability: Apriori can be slow for very large datasets due to its need to generate all possible itemsets.
Data Sparsity: High-dimensional datasets can result in a large number of itemsets that do not meet the minimum support threshold.


Applications
Market Basket Analysis: Identifying frequently co-purchased products in retail.
Recommendation Systems: Suggesting items that are commonly bought together.
Medical Diagnosis: Discovering associations between symptoms and diseases.


Conclusion
Association Rule Mining, and specifically the Apriori algorithm, is a key method in data mining for uncovering interesting relationships between items in large datasets. Its practical applications, especially in retail and marketing, make it a valuable tool in understanding customer behavior and improving business strategies.

"""
