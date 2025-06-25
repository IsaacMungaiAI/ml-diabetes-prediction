# machine learning model for diabetes prediction

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.metrics import ConfusionMatrixDisplay

import joblib

df=pd.read_csv('diabetes.csv')
print(df.head())
print(df.isnull().sum())




#check if there is null values
print(df.isnull().sum())

#correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation heatmap")
plt.show()

#countplot of outcome
sns.countplot(x='Outcome', data=df)
plt.title("Diabetes distribution (0=No, 1=Yes")
plt.show()


#data preprocessing

#features and labels
X=df.drop('Outcome', axis=1)
y=df['Outcome']

#split the dataset
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#normalize the data
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)




#train the machine learning model. Used smote to generate synthentic data to balance the training dataset
smote= SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

param_grid= {
    'n_estimators': [100, 150],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
}

grid_search=GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid, scoring='recall')
grid_search.fit(X_resampled, y_resampled)

#model
best_model = grid_search.best_estimator_
#predict
y_pred_rf=best_model.predict(X_test_scaled)

#evaluate
print("Accuracy: ", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


#visualize the results
ConfusionMatrixDisplay.from_estimator(best_model, X_test_scaled, y_test, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

#show which features mostly impact prediction
importances=best_model.feature_importances_
feature_names= X.columns
feature_importance_df=pd.DataFrame({'Feature':feature_names, 'Importance':importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df.head(10))

# Plot top 10 features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis', hue='Feature', legend=False)
plt.title('Top 10 Most Important Features')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


joblib.dump(best_model, 'diabetes_model2.pkl')
joblib.dump(scaler, 'scaler.pkl')