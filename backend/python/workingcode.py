

#without oversampling with visualization
'''from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
from pymongo import MongoClient
from flask_cors import CORS
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["BrainStrokeData"]
collection = db["cleanedBrainData2"]

# Retrieve data from MongoDB
cursor = collection.find({}, {'_id': 0, 'name': 0, 'email': 0, 'stroke_prediction': 0, 'probability_of_stroke': 0, 'risk_level': 0})
df = pd.DataFrame(list(cursor))

# Assuming df is your DataFrame
X_train = df.drop('stroke', axis=1)
y_train = df['stroke']

# Drop rows with NaN values in the target variable
y_train = y_train.dropna()

# Ensure the number of samples is consistent between X_train and y_train
common_index = X_train.index.intersection(y_train.index)
X_train = X_train.loc[common_index]
y_train = y_train.loc[common_index]

# Perform Univariate Feature Selection using chi-squared test
k = 10
selector = SelectKBest(score_func=chi2, k=k)
X_train_new = selector.fit_transform(X_train, y_train)

# Train the Random Forest model on the data
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train_new, y_train)

# Visualizations applicable to the training phase
# Feature Importance Plot
feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features (Training Phase)')
plt.savefig('feature_importance_training.png')  # Save the plot
plt.close()

# Confusion Matrix Heatmap
y_pred_train = model.predict(X_train_new)
conf_matrix = pd.crosstab(y_train, pd.Series(y_pred_train), rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Training Phase)')
plt.savefig('confusion_matrix_training.png')  # Save the plot
plt.close()

# Classification Report Visualization
report_data = pd.DataFrame.from_dict(classification_report(y_train, y_pred_train, output_dict=True)).iloc[:, :-1]
sns.heatmap(report_data.T, annot=True, cmap='Blues', fmt='.2f')
plt.title('Classification Report (Training Phase)')
plt.savefig('classification_report_training.png')  # Save the plot
plt.close()

# ... (rest of the code)
def predict_stroke(input_data):
    # Create a DataFrame from input data
    user_df = pd.DataFrame([input_data])

    # Drop 'name' and 'email' columns before scaling
    user_df = user_df.drop(['name', 'email'], axis=1, errors='ignore')

    # Select relevant features using the same selector
    user_new = selector.transform(user_df)

    # Make predictions on user input data
    user_pred = model.predict(user_new)

    # Get predicted probabilities for class '1' (stroke)
    user_prob = model.predict_proba(user_new)[:, 1]

    # Define threshold values for risk levels
    high_risk_threshold = 0.8
    moderate_risk_threshold = 0.5
    low_risk_threshold = 0.2

    # Categorize predictions into risk levels
    if user_prob >= high_risk_threshold:
        risk_level = "High Risk"
    elif moderate_risk_threshold <= user_prob < high_risk_threshold:
        risk_level = "Moderate Risk"
    elif low_risk_threshold <= user_prob < moderate_risk_threshold:
        risk_level = "Low Risk"
    else:
        risk_level = "Very Low Risk"

    # Return predictions
    result = {
        "prediction": "Stroke" if user_pred[0] == 1 else 'No Stroke',
        "probability_of_stroke": f"{user_prob[0]:.2%}",
        "risk_level": risk_level
    }

    # Save user data to MongoDB
    input_data['stroke_prediction'] = int(user_pred[0])
    input_data['probability_of_stroke'] = user_prob[0]
    input_data['risk_level'] = risk_level
    collection.insert_one(input_data)

    return result

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        prediction_result = predict_stroke(input_data)
        return jsonify(prediction_result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)'''




#with oversampling with visualization
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from pymongo import MongoClient
from flask_cors import CORS
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Connect to MongoDB
client = MongoClient("mongodb+srv://visheshalag03:8DSgnh73lDWRicC5@stroke-guardian.mkbxrfb.mongodb.net/")
db = client["BrainStrokeData"]
collection = db["cleanedBrainData2"]

# Retrieve data from MongoDB
cursor = collection.find({}, {'_id': 0, 'name': 0, 'email': 0, 'stroke_prediction': 0, 'probability_of_stroke': 0, 'risk_level': 0})
df = pd.DataFrame(list(cursor))

# Assuming df is your DataFrame
X_train = df.drop('stroke', axis=1)
y_train = df['stroke']

# Drop rows with NaN values in the target variable
y_train = y_train.dropna()

# Ensure the number of samples is consistent between X_train and y_train
common_index = X_train.index.intersection(y_train.index)
X_train = X_train.loc[common_index]
y_train = y_train.loc[common_index]

# Perform Univariate Feature Selection using chi-squared test
k = 10
selector = SelectKBest(score_func=chi2, k=k)
X_train_new = selector.fit_transform(X_train, y_train)

# Oversample the minority class
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_new, y_train)

# Train the Random Forest model on resampled data
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Get feature names after feature selection
selected_feature_names = X_train.columns[selector.get_support()]
print("\n", selected_feature_names, "\n")

# Visualizations applicable to the training phase
# Feature Importance Plot
plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
feature_importance = pd.Series(model.feature_importances_, index=selected_feature_names)
feature_importance.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features (Training Phase)')
plt.savefig('feature_importance_training_random.png')  # Save the plot
plt.close()

# Confusion Matrix Heatmap
y_pred_train = model.predict(X_train_resampled)
conf_matrix = pd.crosstab(y_train_resampled, pd.Series(y_pred_train), rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Training Phase)')
plt.savefig('confusion_matrix_training_random.png')  # Save the plot
plt.close()

# Classification Report Visualization
report_data = pd.DataFrame.from_dict(classification_report(y_train_resampled, y_pred_train, output_dict=True)).iloc[:, :-1]
sns.heatmap(report_data.T, annot=True, cmap='Blues', fmt='.2f')
plt.title('Classification Report (Training Phase)')
plt.savefig('classification_report_training_random.png')  # Save the plot
plt.close()

# ... (rest of the code)
def predict_stroke(input_data):
    # Create a DataFrame from input data
    user_df = pd.DataFrame([input_data])

    # Drop 'name' and 'email' columns before scaling
    user_df = user_df.drop(['name', 'email'], axis=1, errors='ignore')

    # Select relevant features using the same selector
    user_new = selector.transform(user_df)

    # Make predictions on user input data
    user_pred = model.predict(user_new)

    # Get predicted probabilities for class '1' (stroke)
    user_prob = model.predict_proba(user_new)[:, 1]

    # Define threshold values for risk levels
    high_risk_threshold = 0.8
    moderate_risk_threshold = 0.5
    low_risk_threshold = 0.2

    # Categorize predictions into risk levels
    if user_prob >= high_risk_threshold:
        risk_level = "High Risk"
    elif moderate_risk_threshold <= user_prob < high_risk_threshold:
        risk_level = "Moderate Risk"
    elif low_risk_threshold <= user_prob < moderate_risk_threshold:
        risk_level = "Low Risk"
    else:
        risk_level = "Very Low Risk"

    # Return predictions
    result = {
        "prediction": "Stroke" if user_pred[0] == 1 else 'No Stroke',
        "probability_of_stroke": f"{user_prob[0]:.2%}",
        "risk_level": risk_level
    }

    # Save user data to MongoDB
    input_data['stroke_prediction'] = int(user_pred[0])
    input_data['probability_of_stroke'] = user_prob[0]
    input_data['risk_level'] = risk_level
    collection.insert_one(input_data)

    return result

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        prediction_result = predict_stroke(input_data)
        return jsonify(prediction_result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))




