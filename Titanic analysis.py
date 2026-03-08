# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("train.csv")

# Display First 5 Rows
print("First 5 Rows of Dataset")
print(df.head())

# Dataset Information
print("\nDataset Information")
print(df.info())

# Check Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Fill Missing Values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin Column (too many missing values)
df.drop('Cabin', axis=1, inplace=True)

# Summary Statistics
print("\nSummary Statistics")
print(df.describe())

# Survival Count Plot
plt.figure()
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Survival by Gender
plt.figure()
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# Survival by Passenger Class
plt.figure()
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Age Distribution
plt.figure()
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution of Passengers")
plt.show()

# Fare Distribution
plt.figure()
sns.histplot(df['Fare'], kde=True)
plt.title("Fare Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Conclusion
print("\nEDA Completed Successfully")
print("Observation: Female passengers and first-class passengers had higher survival rates.")