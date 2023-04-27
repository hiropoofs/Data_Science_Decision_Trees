import pandas as pd
import sklearn.tree as tree
import sklearn.metrics as metric
import sklearn.ensemble as ense
import graphviz

# Load dataset
data = pd.read_csv("CO2 Emissions_Canada.csv")

# Factorize categorical variables
data["Make"], _ = pd.factorize(data["Make"])
data["Model"], _ = pd.factorize(data["Model"])
data["Vehicle Class"], _ = pd.factorize(data["Vehicle Class"])
data["Transmission"], _ = pd.factorize(data["Transmission"])
data["Fuel Type"], fuel_names = pd.factorize(data["Fuel Type"])

# Split data into train and test sets
partition = int(len(data) * 0.7) # 70% partition
train_data, test_data = data[:partition].drop(columns="Fuel Type"), data[partition:].drop(columns="Fuel Type")
train_class, test_class = data[:partition]["Fuel Type"], data[partition:]["Fuel Type"]

# Display fuel type name mapping
print("Fuel type name mapping:", fuel_names.values)

# Define error metric functions
def mae(true, pred):
    return ((true - pred).abs()).sum() / len(true)

# Define function to display accuracy metrics
def display_accuracy(name, true, pred):
    print("_________", name, "_________")
    mse = ((true - pred) ** 2).sum() / len(true)
    print("MSE:", mse)
    print("MAE:", mae(true, pred))
    print("RMSE:", mse ** 0.5)
    print("Confusion matrix:")
    matrix = metric.confusion_matrix(true, pred)
    print(matrix)
    print("Accuracy for each class:")
    print(matrix.diagonal() / matrix.sum(axis=1) * 100)

# Build and render decision tree model
tree_clf = tree.DecisionTreeClassifier(criterion="gini", splitter="best")
tree_model = tree_clf.fit(train_data, train_class)
dot_data = tree.export_graphviz(tree_model, out_file=None, feature_names=train_data.columns, class_names=fuel_names, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("tree")
display_accuracy("Decision Tree (depth=inf)", test_class, tree_model.predict(test_data))

print("help this is the ending of my lifeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")

for depth in [4, 7, 10, 12]:
  tree_clf = tree.DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=depth)
  tree_model = tree_clf.fit(train_data, train_class)
  display_accuracy(f"Decision Tree (depth={depth})", test_class, tree_model.predict(test_data))


# Build forest models with n_estimators = 5 and max_depth values ranging from 1 to 12
mae_min = test_class.max()
tree_min = 0
for i in range(1, 12):
  tree_clf = ense.RandomForestClassifier(n_estimators = 5, max_depth = i)
  tree_model = tree_clf.fit(train_data, train_class)
  MAE = mae(test_class, tree_model.predict(test_data))
  if (MAE < mae_min):
    mae_min = MAE
    tree_min = i
    model_min = tree_model

print("random_forest_best_depth:", tree_min)
display_accuracy("random_forest_best_depth", test_class, model_min.predict(test_data))


# Build forest models with n_estimators = 3:9 and max_depth value 11
mae_min = test_class.max()
for forest_size in range(3, 9):
  tree_clf = ense.RandomForestClassifier(n_estimators = forest_size, max_depth = 11)
  tree_model = tree_clf.fit(train_data, train_class)

  MAE = mae(test_class, tree_model.predict(test_data))
  if (MAE < mae_min):
    mae_min = MAE
    size_min = forest_size
    model_min = tree_model

print("random_forest_best_size:", size_min)
display_accuracy("random_forest_best_size", test_class, model_min.predict(test_data))
