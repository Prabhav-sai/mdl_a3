import itertools

def Apriori(input_list, min_support, min_confidence):
    # convert transactions to sets
    transactions = [set(transaction) for transaction in input_list]
    number_transaction = len(transactions)

    # find all unique items in the dataset
    unique_items = set()
    for transaction in transactions:
        unique_items.update(transaction)
    # unique items is the all items in the dataset

    # find frequency of all unique items
    item_counts = {}
    for item in unique_items:
        item_counts[frozenset([item])] = 0  # Changed: use frozenset as key
    
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1  # Changed: use frozenset as key
    
    # filter the items by min_support
    L1 = {item: item_counts[item]/number_transaction for item in item_counts if item_counts[item]/number_transaction >= min_support}
    # print("L1:", L1)

    # store all frequent itemsets by size
    L = [L1]
    k = 2

    # print("L1:", L1.keys())

    while L[-1]:        # L[-1] is the last frequent itemset
        Ck = generate_candidates(L[-1].keys(), k)
        # print("Ck:", Ck)
        # count support for each candidate
        count = {c: 0 for c in Ck}

        # for each transaction, find candidates that are subsets
        for transaction in transactions:
            Dt = {c for c in Ck if c.issubset(transaction)}
            for c in Dt:
                count[c] += 1
        
        # filter candidates by min_support
        itemset_counts = {}
        for c in Ck:
            support = count[c] / number_transaction
            if support >= min_support:
                itemset_counts[c] = support
        
        if not itemset_counts:
            break

        L.append(itemset_counts)
        k += 1
    
    rules = []
    
    all_frequent_itemsets = {}
    for level_itemsets in L:
        all_frequent_itemsets.update(level_itemsets)
    # print("all_frequent_itemsets:", all_frequent_itemsets)

    for k in range(1,len(L)):
        for itemset, support in L[k].items():
            if len(itemset) < 2:
                continue
            
            for i in range(1, len(itemset)):
                # generate all possible antecedents and consequents
                for antec in itertools.combinations(itemset, i):
                    antecedent = frozenset(antec)
                    consequent = itemset - antecedent
                    # calculate support and confidence
                    support_antecedent = all_frequent_itemsets[antecedent]

                    confidence = support / support_antecedent

                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))
    
    for antec, consequent, confidence in rules:
        print(f"{', '.join(sorted(set(antec)))} -> {', '.join(sorted(set(consequent)))}")

    # return L , rules
    return

def generate_candidates(prev_frequent, k):
    """Generate candidate itemsets of size k from frequent itemsets of size k-1."""
    candidates = set()
    prev_list = list(prev_frequent)     # convert to list from set
    # print("prev_list:", prev_list)

    for i, j in itertools.combinations(prev_list, 2):
        # join step: merge two itemsets if they have k-2 items in common
        union = i.union(j)
        if len(union) == k:
            all_subsets_frequent = True
            for subset in itertools.combinations(union, k-1):
                if frozenset(subset) not in prev_frequent:  # Changed: use frozenset
                    all_subsets_frequent = False
                    break
            
            if all_subsets_frequent:
                candidates.add(frozenset(union))
    
    # print("candidates:", candidates)
    return candidates


# input_list = [
#     ['A', 'B', 'C'],
#     ['D', 'E'],
#     ['A', 'D', 'F'],
#     ['E', 'F'],
#     ['B', 'D', 'E'],
#     ['A', 'D'],
#     ['A', 'B', 'E', 'D'],
#     ['C', 'F'],
#     ['A', 'B', 'D'],
#     ['D', 'E']
# ]
# min_support = 0.2
# min_confidence = 0.6

# Apriori(input_list, min_support, min_confidence)

# L , rules = apriori(input_list, min_support, min_confidence)

# print("frequent itemsets:")
# for i, itemset in enumerate(L):
#     print(f"L{i+1}: {itemset}")

# print("rules:")
# for antec, consequent, confidence in rules:
#     print(f"Rule: {set(antec)} -> {set(consequent)}, Confidence: {confidence:.2f}")

# print(L)