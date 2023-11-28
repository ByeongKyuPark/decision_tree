# PROJECT_EXTRA : decision tree
# ByoengKyu Park(byeonggyu.park)
import numpy as np
import pandas as pd
from graphviz import Digraph
import random

#---------------------------------------------------------------------HELPER FUNCTIONS
# calc the entropy
def calc_entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


# calc the info gain (for ties, pick randomly)
def calc_IG(data, split_attribute_name, target_name):
    total_entropy = calc_entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*calc_entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


# id3 algorithm
def ID3(data, originaldata, features, target_attribute_name, parent_node_class = None, depth=0):
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
        item_values = [calc_IG(data, feature, target_attribute_name) for feature in features] # Return the information gain values for the features in the dataset
        
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
def plot_tree(graph, tree, parent_name, node_path):
    # Base case: if the tree is a tuple, we have reached a leaf
    if isinstance(tree, tuple):
        survival_status = "Survived" if tree[0] == 1 else "Not Survived"
        node_label = f"Leaf: {survival_status}\nPassengerId: {tree[1]}"
        leaf_name = f'leaf_{node_path}'
        graph.node(leaf_name, node_label, shape='box')
        graph.edge(parent_name, leaf_name)
        return

    # Recursive case: iterate through the branches of the tree
    for split_feature, subtree in tree.items():
        # Use the unique node_path and feature name to create a unique node name
        feature_name = str(split_feature)
        node_name = f'{node_path}_{feature_name}'
        # Create the node for this feature split
        graph.node(node_name, feature_name)
        # Connect the parent node to this node
        if parent_name:
            graph.edge(parent_name, node_name)
        
        for feature_value, child_tree in subtree.items():
            # Convert numeric values to their respective categories
            if split_feature == "Sex":
                value_str = "Male" if feature_value == 0 else "Female"
            elif split_feature == "Age":
                age_categories = {0: "Child", 1: "Young Adult", 2: "Adult", 3: "Senior"}
                value_str = age_categories.get(feature_value, str(feature_value))
            elif split_feature == "Fare":
                fare_categories = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
                value_str = fare_categories.get(feature_value, str(feature_value))
            else:
                value_str = str(feature_value)
            
            child_path = f'{node_name}_{value_str}'
            graph.node(child_path, value_str)
            graph.edge(node_name, child_path)
            plot_tree(graph, child_tree, child_path, child_path)
#--------------------------------------------------------------------- MAIN
# (0) Load the train dataset
data_path = 'titanic/train.csv'
titanic_data = pd.read_csv(data_path)

# Set PassengerId as the index
titanic_data.set_index('PassengerId', inplace=True)

# check
titanic_data.head()
print(titanic_data.shape)

# Dropping non-relevant & non-numeric attributes
titanic_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Manually encode the 'Sex' column
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})

# Categorize 'Age' into 'Child', 'Young Adult', 'Adult', 'Senior'
age_bins = [0, 12, 18, 60, 100]  # 0-11: Child, 12-17: Young Adult, 18-59: Adult, 60-100: Senior
age_labels = [0, 1, 2, 3]  # 0: Child, 1: Young Adult, 2: Adult, 3: Senior
titanic_data['Age'] = pd.cut(titanic_data['Age'], bins=age_bins, labels=age_labels, right=False)

# Categorize 'Fare' into 'Low', 'Medium', 'High', 'Very High'
fare_bins = [-1, 10, 50, 100, 600]  # -1-9: Low, 10-49: Medium, 50-99: High, 100-600: Very High
fare_labels = [0, 1, 2, 3]  # 0: Low, 1: Medium, 2: High, 3: Very High
titanic_data['Fare'] = pd.cut(titanic_data['Fare'], bins=fare_bins, labels=fare_labels, right=False)

# Now, we encode it into numerical categories
embarked_mapping = {label: idx for idx, label in enumerate(titanic_data['Embarked'].unique())}
titanic_data['Embarked'] = titanic_data['Embarked'].map(embarked_mapping)

# (1) build the tree
titanic_features = titanic_data.columns.drop('Survived')  # All columns except 'Survived'
titanic_tree = ID3(titanic_data, titanic_data, titanic_features, target_attribute_name="Survived")

# init
dot = Digraph(comment='Titanic ecision Tree')

# (2-1) plot the decision tree
plot_tree(dot, titanic_tree, '', 'root')

# (2-2) Save and render the graph to an output file
dot.render('decision_tree_project_titanic', view=True, format='png')
