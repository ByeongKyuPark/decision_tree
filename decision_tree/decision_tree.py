# HW7 : decision tree
# ByoengKyu Park(byeonggyu.park)
import numpy as np
import pandas as pd
from graphviz import Digraph
import random

#---------------------------------------------------------------------DATA
# the dataset
data = pd.DataFrame({
	'Age': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2],
	'Prescription': [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2],
	'Astigmatic': [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1],
	'Tear': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
	'Label': [3, 2, 3, 1, 3, 2, 3, 1, 3, 2, 3, 1, 3, 2, 3, 3, 3, 3, 3, 1, 3, 2, 3, 2]
})

#---------------------------------------------------------------------HELPER FUNCTIONS
# calc the entropy
def calc_entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


# calc the info gain (for ties, pick randomly)
def cacl_IG(data, split_attribute_name, target_name="Label"):
    total_entropy = calc_entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*calc_entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

# id3 algorithm
def ID3(data, originaldata, features, target_attribute_name="Label", parent_node_class = None, depth=0):
    # if all in the subset have the same label, return the label with the indices
    if len(np.unique(data[target_attribute_name])) <= 1:
        return (np.unique(data[target_attribute_name])[0], data.index.tolist())
    
    # for empty subset, returns the mode label of the parent subset
    elif data.empty:
        return (parent_node_class if parent_node_class is not None else np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])], [])
        
    # no more attributes to consider, return the mode label in the current subset
    elif len(features) == 0:
        return (np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])], data.index.tolist())
        
    # building branches
    else:
        # the default value for this node is the mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        # decide the feature which yields the highest IG (=lowest entropy)
        item_values = [cacl_IG(data, feature, target_attribute_name) for feature in features] # Return the information gain values for the features in the dataset
        
        # picks a random attribute for ties
        max_gain = max(item_values)
        best_features = [features[i] for i, gain in enumerate(item_values) if gain == max_gain]
        best_feature = random.choice(best_features) if len(best_features) > 1 else best_features[0]
        
        tree = {best_feature:{}}
        
        # extract remaining features
        features = [i for i in features if i != best_feature]
        
        # Grow a branch 
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            
            # recursive call for subtrees
            subtree = ID3(sub_data, originaldata, features, target_attribute_name, parent_node_class, depth+1)
            
            # add the subtree to the tree under the root node
            tree[best_feature][value] = subtree
            
        return(tree)

#---------------------------------------------------------------------PLOTTING FUNCTION
def plot_tree(graph, tree, parent_name, counter=0):
    # Base case
    if type(tree) == tuple:
        node_label = f"Leaf: {tree[0]}\nRows: {tree[1]}"
        graph.node(f'leaf{counter}', node_label, shape='box')
        graph.edge(parent_name, f'leaf{counter}')
        return counter + 1

    # Recursive case
    for split_feature, subtree in tree.items():
        graph.node(split_feature, split_feature)
        if parent_name:
            graph.edge(parent_name, split_feature)
        
        for feature_value, child_tree in subtree.items():
            child_name = f"{split_feature}_{feature_value}"
            graph.node(child_name, f"{feature_value}")
            graph.edge(split_feature, child_name)
            counter = plot_tree(graph, child_tree, child_name, counter)

    return counter
#--------------------------------------------------------------------- MAIN
# (1) build the tree
tree = ID3(data, data, data.columns[:-1])
print(tree)

# init
dot = Digraph(comment='Decision Tree')

# (2-1) plot the decision tree
plot_tree(dot, tree, '', 0)

# (2-2) Save and render the graph to an output file
dot.render('C:/math345/decision_tree', view=True, format='png')

