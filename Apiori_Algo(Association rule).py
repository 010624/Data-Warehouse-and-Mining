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

