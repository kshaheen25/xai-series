# %% Imports
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import shap

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
print(X_train.shape)
print(X_test.shape)
feature_names = X_test.columns.tolist()
print(len(feature_names))

# %% Fit blackbox model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Create SHAP explainer
explainer = shap.TreeExplainer(rf)
# Calculate shapley values for test data
start_index = 1
end_index = 2
shap_values = explainer.shap_values(X_test)
print(X_test[start_index:end_index])
print(np.array(shap_values).shape) 
print("SHAP values shape:", np.array(shap_values).shape)
print("Features shape:", X_test.iloc[0].shape)
print("Number of feature names:", len(feature_names))



# %% Investigating the values (classification problem)
# class 0 = contribution to class 1
# class 1 = contribution to class 2
print(shap_values[0].shape)
shap_values
print(explainer.expected_value[1])



# %% >> Visualize local predictions
shap.initjs()
# Force plot
class_index = 0  
sample_index = 0
output_value_index = 0
prediction = rf.predict(X_test[start_index:end_index])[0]
print(f"The RF predicted: {prediction}")

sample_index = 0  # Index of the sample to plot

shap.force_plot(
    explainer.expected_value[output_value_index],
    shap_values[sample_index, :, output_value_index],  # SHAP values for the selected sample and output value
    X_test.iloc[sample_index],  # Feature values for the selected sample
    feature_names=X_test.columns.tolist(),
    show=False
)



# %% >> Visualize global features
# Feature summary
shap.force_plot(
    explainer.expected_value[output_value_index],  # Base value for the output
    shap_values[:, :, output_value_index].mean(axis=0),  # Average SHAP values across all samples for the first output value
    X_test.mean(axis=0),  # Average feature values across all samples
    feature_names=X_test.columns.tolist(),  # List of feature names
    show=False  # Set to False for non-interactive environments
)

shap.summary_plot(shap_values, X_test)


# %%
# Choosing the first sample's SHAP values for plotting
import shap

# Assuming 'model' is your trained RandomForest or any other tree-based model
# Assuming 'X_test' is your testing dataset prepared as a Pandas DataFrame

# Initialize the SHAP explainer on the model
explainer = shap.TreeExplainer(rf)

# Calculate SHAP values for all instances in the test set
shap_values = explainer.shap_values(X_test)

# Select a specific sample to explain
sample_index = 0  # Change this as needed to explore other instances

# Create an Explanation object for the selected sample
explanation = shap.Explanation(values=shap_values[0][sample_index],  # SHAP values for the first class, adjust index for multi-class
                               base_values=explainer.expected_value[0],  # Base (expected) value for the output
                               data=X_test.iloc[sample_index].values,  # Feature values for the instance
                               feature_names=X_test.columns.tolist())  # Names of the features

# Generate and display the waterfall plot
shap.plots.waterfall(explanation, show=True)  # Set show=True to display the plot inline if using Jupyter Notebook



# %%
# Setting up the feature names, assuming X_test is a DataFrame with proper columns
# Use show=True to display the plot inline if using a Jupyter Notebook

import numpy as np
import shap

# Assuming 'model' is your trained model and 'X_test' is your test dataset prepared as a Pandas DataFrame

# Initialize the SHAP explainer on the model
explainer = shap.TreeExplainer(rf)

# Calculate SHAP values for the test set (assuming binary classification and interested in class 0)
shap_values = explainer.shap_values(X_test)[0]



# Calculate mean SHAP values across all samples for class 0
mean_shap_values = np.mean(shap_values, axis=0)

# Create an aggregated Explanation object using mean SHAP values
aggregated_explanation = shap.Explanation(values=mean_shap_values,
                                          base_values=explainer.expected_value[0],
                                          data=np.mean(X_test, axis=0),  # Mean of features across all samples
                                          feature_names=X_test.columns.tolist())

# Generate and display the waterfall plot for the aggregated Explanation
shap.plots.waterfall(aggregated_explanation, show=True)  # Set show=True to display the plot inline if using Jupyter Notebook


# %%
shap.plots.waterfall(aggregated_explanation, max_display=21, show=True)


# %%
