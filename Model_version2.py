import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

dataset = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
print(dataset)

le = LabelEncoder()

for i in dataset.columns[ : ] :
    dataset[i] = le.fit_transform(dataset[i])

X = dataset.drop(['Diabetes_binary', 'CholCheck', 'Stroke', 'HeartDiseaseorAttack', 'AnyHealthcare', 'NoDocbcCost', 'MentHlth', 'PhysHlth', 'Income'],axis="columns")
y = dataset['Diabetes_binary']

X_DT = X [['HighBP', 'HighBS', 'BMI', 'Smoker', 'PhysActivity',
           'Fruits', 'Veggies', 'HvyAlcoholConsump', 'GenHlth',
           'DiffWalk', 'Sex', 'Age', 'Education']]
X_train, X_test, y_train, y_test = train_test_split(X_DT, y, test_size = 0.3,random_state=42,shuffle=True)

tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
DT= tree.score(X_test, y_test)

from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_test, y_pred) * 100, "%")
print(classification_report(y_test, y_pred))


with open('model.pkl','wb') as model:
    pickle.dump(tree,model)