import warnings
import  os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
df.head()

# แปลงข้อมูลจาก float64ทศนิยม เป็นชนิด int64จำนวนเต็ม
df['Diabetes_binary'] = df['Diabetes_binary'].astype('int')
df['HighBP'] = df['HighBP'].astype('int')
df['HighBS'] = df['HighBS'].astype('int')
df['CholCheck'] = df['CholCheck'].astype('int')
df['BMI'] = df['BMI'].astype('int')
df['Smoker'] = df['Smoker'].astype('int')
df['Stroke'] = df['Stroke'].astype('int')
df['HeartDiseaseorAttack'] = df['HeartDiseaseorAttack'].astype('int')
df['PhysActivity'] = df['PhysActivity'].astype('int')
df['Fruits'] = df['Fruits'].astype('int')
df['Veggies'] = df['Veggies'].astype('int')
df['HvyAlcoholConsump'] = df['HvyAlcoholConsump'].astype('int')
df['AnyHealthcare'] = df['AnyHealthcare'].astype('int')
df['NoDocbcCost'] = df['NoDocbcCost'].astype('int')
df['GenHlth'] = df['GenHlth'].astype('int')
df['MentHlth'] = df['MentHlth'].astype('int')
df['PhysHlth'] = df['PhysHlth'].astype('int')
df['DiffWalk'] = df['DiffWalk'].astype('int')
df['Sex'] = df['Sex'].astype('int')
df['Age'] = df['Age'].astype('int')
df['Education'] = df['Education'].astype('int')
df['Income'] = df['Income'].astype('int')
df.head()

df['Diabetes_binary'].value_counts()

# แสดงสรุปข้อมูลว่ามีกี่แถว, Missing value เท่าไหร่, แต่ละคอลัมน์เป็น Data Type อะไรบ้าง"""
df.info()

# แสดงจำนสนค่าที่สูญหายทั้งหมดในแต่ละคอลัมน์"""
df.isna().sum()

# เช็ค Summary ใจความสำคัญของแต่ละคอลัมน์ (count, min, max, mean)"""
df.describe()

# ปรับสเกลข้อมูลให้สมดุลกัน"""
df.corr()

# **Correlation** (เมทริกซ์สหสัมพันธ์) จะมีค่าอยู่ระหว่าง -1.0 ถึง +1.0 ซึ่งหากมีค่าใกล้ -1.0 นั้นหมายความว่าตัวแปรทั้งสองตัวมีความสัมพันธ์กันอย่างมากในเชิงตรงกันข้าม
# หากมีค่าใกล้ +1.0 นั้นหมายความว่า ตัวแปรทั้งสองมีความสัมพันธ์กันอย่างมากในทิศทางเดียวกัน และหากมีค่าเป็น 0 นั้นหมายความว่า ตัวแปรทั้งสองตัวไม่มีความสัมพันธ์ต่อกัน
cor_mat = df.corr()
fig, ax = plt.subplots(figsize=(18, 12))
sns.heatmap(cor_mat, annot=True, linewidths=0.2, fmt=".3f")
plt.title('Pearson Correlation')

# แบ่งข้อมูลเป็น X,y โดยเราเลือกทุก Feature มาใช้กับโมเดล"""

feature = ['HighBP', 'HighBS', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity',
           'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',
           'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
X = df[feature]
y = df['Diabetes_binary']

# ทำการ Scale ข้อมูลในตัวแปร X ทั้งหมด
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(X)
print(X)

# จากนั้นจึงค่อยนำข้อมูลไปแยกเป็นชุดฝึกและชุดทดสอบ
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# จากนั้นทำการหา hyper parameter และ parameter ที่เหมาสมกับ dataset ของเรา
# โดย Decision tree จะมีสิ่งที่จะต้องปรับหลักๆคือ max_depth จำนวนชั้นของต้นไม้ ถ้า max_depth เป็น 3 ชั้นความลึกของต้นไม้เราจะไม่เกิน 3
# นั้นเองเราจะทำการลองกับค่า max_depth 1–10 ว่าค่าไหนจะดีที่สุด ก็จะได้ว่า max_depth = 7
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': np.arange(1, 10), 'criterion': ['entropy', 'gint']}
tree = GridSearchCV(DecisionTreeClassifier(), param_grid)
tree.fit(X_train, y_train)
tree.best_estimator_

# จากนั้นก็นำโมเดลไปทดสอบกับข้อมูลชุด test ต่อได้เลย
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
DT= tree.score(X_test, y_test)



from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_test, y_pred) * 100, "%")
print(classification_report(y_test, y_pred))

model_comp = pd.DataFrame({'Model': ['DecisionTree'], 'Accuracy': [DT* 100]})
print(model_comp)

# เราก็จะทำ decision tree ให้ออกมาเป็นรูปภาพเพื่ออธิบายว่าทำไมถึงตัดสินใจว่าข้อมูลนั้น เป็นเบาหวานหรือไม่เป็นเบาหวานด้วย export_graphviz
from sklearn.tree import export_graphviz

# จากนั้นให้เราใส่ tree ที่ทำการเทรนลงไปใน export_graphviz โดยใส่ feature_name กับ class_names จากนั้นก็ปริ้น tree ออกมาได้เลย
tree_dot = export_graphviz \
    (tree,
     out_file=None,  # or out_file=”iris_tree.dot”
     feature_names=feature,
     class_names='Diabetes_binary',
     rounded=True,
     filled=True
     )
print(tree_dot)

# สร้าง decistion tree จากข้อความที่รันได้ด้านบนออกมาเป็นรูป เราก็จะได้ decision tree ออกมา วิธีการใช้งานคือดูตามเงื่อนไขในแต่ละชั้น
# ให้เราก็อปข้อความทั้งหมดไปลงที่เว็บ http://www.webgraphviz.com/
# ดูว่าเราตรงกับเงื่อนไขไหน เลื่อนลงมาสุดจนเงื่อนไขสุดท้าย เราก็จะรู้ว่าเราตรงกับ class ไหน

# โค้ดด้านล่างนี้แสดงรูปภาพโมเดล แต่ใช้ได้ใน google colab
# from graphviz import Source
# from IPython.display import SVG
# from IPython.display import display
# graph = Source(export_graphviz(tree, out_file=None,  filled=True, rounded=True, special_characters=True,feature_names = feature,class_names='Diabetes_binary'))
# display(SVG(graph.pipe(format='svg')))
# print(display(SVG(graph.pipe(format='svg'))))


# คำอธิบายภาพด้านบน
# **samples** คือจำนวนรายการข้อมูลที่เข้ากันได้กับ Node นั้น ดังนั้น เมื่อการตัดสินใจเคลื่อนลงไปตามความลึกของต้นไม้ จำนวน samples ของ Node
# ในแต่ละชั้นจะมีแนวโน้มที่จะลดลงเรื่อยๆgini บ่งชี้ความ "บริสุทธิ์" ของ Node โดย gini = 0 หมายความว่าข้อมูลทุกรายการใน Node นั้นอยู่ใน Class เดียวกัน
# ส่วน gini = 0.5 ก็แปลว่ารายการข้อมูลใน Node นั้นอยู่ใน 2 Class เท่าๆ กัน โดยแสดงผ่าน value เช่น value = 7062, 6235 ใน Child node ด้านขวาของ Root node
# แปลว่า จากข้อมูล 13297 รายการที่เข้าเงื่อนไข Node นี้ มี 7062 รายการที่อยู่ใน Class 0 และ 6235 รายการอยู่ใน Class 1 โดยถ้าหยุดพิจารณาที่ขั้นนี้ ก็จะถือว่าข้อมูลที่เข้าเงื่อนไขของ
# Node นี้อยู่ใน Class 0

# confusion matrix เมทริกซ์ความสับสน
from sklearn import metrics

matrix = metrics.confusion_matrix(y_test, y_pred)
print(matrix)

# heatmap matrix
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, linewidths=0.1, fmt=".0f", cmap='viridis')
plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Actual")

# รายงานการจัดหมวดหมู่
report = metrics.classification_report(y_test, y_pred)
print(report)

# สร้างภาพข้อมูล  transform data
df.Diabetes_binary[df['Diabetes_binary'] == 0] = 'No Diabetes'
df.Diabetes_binary[df['Diabetes_binary'] == 1] = 'Diabetes'

df.HighBP[df['HighBP'] == 0] = 'No High'
df.HighBP[df['HighBP'] == 1] = 'High BP'

df.HighBS[df['HighBS'] == 0] = 'No High Blood Sugar'
df.HighBS[df['HighBS'] == 1] = 'High Blood Sugarl'

df.CholCheck[df['CholCheck'] == 0] = 'No Cholesterol Check in 5 Years'
df.CholCheck[df['CholCheck'] == 1] = 'Cholesterol Check in 5 Years'

df.Smoker[df['Smoker'] == 0] = 'No'
df.Smoker[df['Smoker'] == 1] = 'Yes'

df.Stroke[df['Stroke'] == 0] = 'No'
df.Stroke[df['Stroke'] == 1] = 'Yes'

df.HeartDiseaseorAttack[df['HeartDiseaseorAttack'] == 0] = 'No'
df.HeartDiseaseorAttack[df['HeartDiseaseorAttack'] == 1] = 'Yes'

df.PhysActivity[df['PhysActivity'] == 0] = 'No'
df.PhysActivity[df['PhysActivity'] == 1] = 'Yes'

df.Fruits[df['Fruits'] == 0] = 'No'
df.Fruits[df['Fruits'] == 1] = 'Yes'

df.Veggies[df['Veggies'] == 0] = 'No'
df.Veggies[df['Veggies'] == 1] = 'Yes'

df.HvyAlcoholConsump[df['HvyAlcoholConsump'] == 0] = 'No'
df.HvyAlcoholConsump[df['HvyAlcoholConsump'] == 1] = 'Yes'

df.AnyHealthcare[df['AnyHealthcare'] == 0] = 'No'
df.AnyHealthcare[df['AnyHealthcare'] == 1] = 'Yes'

df.NoDocbcCost[df['NoDocbcCost'] == 0] = 'No'
df.NoDocbcCost[df['NoDocbcCost'] == 1] = 'Yes'

df.GenHlth[df['GenHlth'] == 1] = 'Excellent'
df.GenHlth[df['GenHlth'] == 2] = 'Very Good'
df.GenHlth[df['GenHlth'] == 3] = 'Good'
df.GenHlth[df['GenHlth'] == 4] = 'Fair'
df.GenHlth[df['GenHlth'] == 5] = 'Poor'

df.DiffWalk[df['DiffWalk'] == 0] = 'No'
df.DiffWalk[df['DiffWalk'] == 1] = 'Yes'

df.Sex[df['Sex'] == 0] = 'Female'
df.Sex[df['Sex'] == 1] = 'Male'

# df.Age[df['Age'] == 1] = '18-24'
# df.Age[df['Age'] == 2] = '25-29'
# df.Age[df['Age'] == 3] = '30-34'
# df.Age[df['Age'] == 4] = '35-39'
# df.Age[df['Age'] == 5] = '40-44'
# df.Age[df['Age'] == 6] = '45-49'
# df.Age[df['Age'] == 7] = '50-54'
# df.Age[df['Age'] == 8] = '55-59'
# df.Age[df['Age'] == 9] = '60-64'
# df.Age[df['Age'] == 10] = '65-69'
# df.Age[df['Age'] == 11] = '70-74'
# df.Age[df['Age'] == 12] = '75-79'
# df.Age[df['Age'] == 13] = '80+'

df.Education[df['Education'] == 1] = 'Never Attended School'
df.Education[df['Education'] == 2] = 'Elementary'
df.Education[df['Education'] == 3] = 'Junior High School'
df.Education[df['Education'] == 4] = 'Senior High School'
df.Education[df['Education'] == 5] = 'Undergraduate Degree'
df.Education[df['Education'] == 6] = 'Magister'

df.Income[df['Income'] == 1] = 'Less Than $10,000'
df.Income[df['Income'] == 2] = 'Less Than $10,000'
df.Income[df['Income'] == 3] = 'Less Than $10,000'
df.Income[df['Income'] == 4] = 'Less Than $10,000'
df.Income[df['Income'] == 5] = 'Less Than $35,000'
df.Income[df['Income'] == 6] = 'Less Than $35,000'
df.Income[df['Income'] == 7] = 'Less Than $35,000'
df.Income[df['Income'] == 8] = '$75,000 or More'
df.head()

df.Diabetes_binary.value_counts()
print(df.Diabetes_binary.value_counts())

plt.figure(figsize=(8, 6))
sns.countplot(df['Diabetes_binary'])
plt.title("Diabetes Status")

# group diabetes status & BP
diabetes_bp = df.groupby(['Diabetes_binary', 'HighBP']).size().reset_index(name='Count')
print(diabetes_bp)

# visualize diabetes status ~ BP
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='HighBP', data=diabetes_bp, palette='Set1')
plt.title("Dibaetes Status ~ BP")

# group diabetes status & cholesterol status
diabetes_chol = df.groupby(['Diabetes_binary', 'HighBS']).size().reset_index(name='Count')
print(diabetes_chol)

# visualize diabetes status ~ cholesterol status
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='HighBS', data=diabetes_chol, palette='Set2')
plt.title("Dibaetes Status ~ Cholesterol Status")

# group diabetes status & cholesterol check
diabetes_check = df.groupby(['Diabetes_binary', 'CholCheck']).size().reset_index(name='Count')
print(diabetes_check)

# visualize diabetes status ~ cholesterol check
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='CholCheck', data=diabetes_check)
plt.title("Dibaetes Status ~ Cholesterol Check")

# visualize diabetes status ~ BMI
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Diabetes_binary', y='BMI', palette='Set1')
plt.title("Dibaetes Status ~ BMI")

# group diabetes status & smoker status
diabetes_smoker = df.groupby(['Diabetes_binary', 'Smoker']).size().reset_index(name='Count')
print(diabetes_smoker)

# visualize diabetes status ~ smoker status
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='Smoker', data=diabetes_smoker, palette='Set2')
plt.title("Dibaetes Status ~ Smoker Status")

# group diabetes status & stroke status
diabetes_stroke = df.groupby(['Diabetes_binary', 'Stroke']).size().reset_index(name='Count')
print(diabetes_stroke)

# visualize diabetes status ~ stroke status
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='Stroke', data=diabetes_stroke, palette='Set1')
plt.title("Dibaetes Status ~ Stroke Status")

# group diabetes status & heart diseaseor attack
diabetes_heart = df.groupby(['Diabetes_binary', 'HeartDiseaseorAttack']).size().reset_index(name='Count')
print(diabetes_heart)

# visualize diabetes status ~ heart diseaseor attack
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='HeartDiseaseorAttack', data=diabetes_heart, palette='Set2')
plt.title("Dibaetes Status ~ Heart Diseaseor Attack")

# group diabetes status & physical activity
diabetes_physical = df.groupby(['Diabetes_binary', 'PhysActivity']).size().reset_index(name='Count')
print(diabetes_physical)

# visualize diabetes status ~ physical activity
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='PhysActivity', data=diabetes_physical)
plt.title("Dibaetes Status ~ Physical Activity")

# group diabetes status & fruits
diabetes_fruit = df.groupby(['Diabetes_binary', 'Fruits']).size().reset_index(name='Count')
print(diabetes_fruit)

# visualize diabetes status ~ fruits
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='Fruits', data=diabetes_fruit, palette='Set1')
plt.title("Dibaetes Status ~ Fruits")

# group diabetes status & veggies
diabetes_veggies = df.groupby(['Diabetes_binary', 'Veggies']).size().reset_index(name='Count')
print(diabetes_veggies)

# visualize diabetes status ~ veggies
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='Veggies', data=diabetes_veggies, palette='Set2')
plt.title("Dibaetes Status ~ Veggies")

# group diabetes status & HvyAlcoholConsump
diabetes_alcohol = df.groupby(['Diabetes_binary', 'HvyAlcoholConsump']).size().reset_index(name='Count')
print(diabetes_alcohol)

# visualize diabetes status ~ HvyAlcoholConsump
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='HvyAlcoholConsump', data=diabetes_alcohol)
plt.title("Dibaetes Status ~ Alcohol Consumption")

# group diabetes status & AnyHealthcare
diabetes_healthcare = df.groupby(['Diabetes_binary', 'AnyHealthcare']).size().reset_index(name='Count')
print(diabetes_healthcare)

# visualize diabetes status ~ AnyHealthcare
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='AnyHealthcare', data=diabetes_healthcare, palette='Set1')
plt.title("Dibaetes Status ~ Healthcare")

# group diabetes status & doctor cost
diabetes_NoDocbcCost = df.groupby(['Diabetes_binary', 'NoDocbcCost']).size().reset_index(name='Count')
print(diabetes_NoDocbcCost)

# visualize diabetes status ~ doctor cost
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='NoDocbcCost', data=diabetes_NoDocbcCost, palette='Set2')
plt.title("Dibaetes Status ~ Doctor Cost")

# group diabetes status & general health
diabetes_general = df.groupby(['Diabetes_binary', 'GenHlth']).size().reset_index(name='Count')
print(diabetes_general)

# visualize diabetes status ~ general health
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='GenHlth', data=diabetes_general)
plt.title("Dibaetes Status ~ General Health")

# visualize diabetes status ~ mental health
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Diabetes_binary', y='MentHlth', palette='Set1')
plt.title("Dibaetes Status ~ Mental Health")

# visualize diabetes status ~ physical health
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Diabetes_binary', y='PhysHlth', palette='Set2')
plt.title("Dibaetes Status ~ Physical Health")

# group diabetes status & difficulty walking
diabetes_walk = df.groupby(['Diabetes_binary', 'DiffWalk']).size().reset_index(name='Count')

# visualize diabetes status ~ difficulty walking
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='DiffWalk', data=diabetes_walk)
plt.title("Dibaetes Status ~ Difficulty Walking")

# group diabetes status & gender
diabetes_sex = df.groupby(['Diabetes_binary', 'Sex']).size().reset_index(name='Count')
print(diabetes_sex)

# visualize diabetes status ~ gender
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='Sex', data=diabetes_sex, palette='Set1')
plt.title("Dibaetes Status ~ Gender")

# visualize diabetes status ~ age
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Diabetes_binary', y='Age', palette='Set2')
plt.title("Dibaetes Status ~ Age")

# group diabetes status & education
diabetes_education = df.groupby(['Diabetes_binary', 'Education']).size().reset_index(name='Count')
print(diabetes_education)

# visualize diabetes status ~ education
plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='Education', data=diabetes_education)
plt.title("Dibaetes Status ~ Education")

# group diabetes status & income
diabetes_income = df.groupby(['Diabetes_binary', 'Income']).size().reset_index(name='Count')
print(diabetes_income)

plt.figure(figsize=(8, 6))
sns.barplot(x='Diabetes_binary', y='Count', hue='Income', data=diabetes_income, palette='Set1')
plt.title("Dibaetes Status ~ Income")

# ทำนายผล
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
feat = scaler.transform(X)
print(feat)

user_input = input("Enter the values one by one")  # ป้อนค่าทีละตัว
user_input = user_input.split(",")

for i in range(len(user_input)):
    # convert each item to int type แปลงแต่ละรายการเป็นประเภท int
    user_input[i] = float(user_input[i])

user_input = np.array(user_input)
user_input = user_input.reshape(1, -1)
user_input = scaler.transform(user_input)
predic = tree.predict(user_input)
if (predic[0] == 0):
    print("You are healthier and less likely to get diabetes! ")  # คุณมีสุขภาพดีและมีโอกาสน้อยที่จะเป็นโรคเบาหวาน!
else:
    print("Warning! Chances of getting diabetes! ")  # คำเตือน! มีโอกาสเป็นโรคเบาหวาน
