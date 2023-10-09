# Implementation-of-Linear-Regression-Using-Gradient-Descent
# AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student. Equipments Required:

Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
# Algorithm
# Steps involved
1.Data Preparation: The first step is to prepare the data for the model. This involves cleaning the data, handling missing values and outliers, and transforming the data into a suitable format for the model.

2.Split the data: Split the data into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate the model's performance.

3.Define the model: The next step is to define the logistic regression model. This involves selecting the appropriate features, specifying the regularization parameter, and defining the loss function.

4.Train the model: Train the model using the training data. This involves minimizing the loss function by adjusting the model's parameters.

5.Evaluate the model: Evaluate the model's performance using the testing data. This involves calculating the model's accuracy, precision, recall, and F1 score.

6.Tune the model: If the model's performance is not satisfactory, you can tune the model by adjusting the regularization parameter, selecting different features, or using a different algorithm.

7.Predict new data: Once the model is trained and tuned, you can use it to predict new data. This involves applying the model to the new data and obtaining the predicted outcomes.

8.Interpret the results: Finally, you can interpret the model's results to gain insight into the relationship between the input variables and the output variable. This can help you understand the factors that influence the outcome and make informed decisions based on the results.

# Program:
```
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
# Output:
# 1.Placement Data
![image](https://github.com/niveshaprabu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122986499/b4a1dda6-f02f-434b-a57b-9f945f4bfc47)


# 2.Salary Data
![image](https://github.com/niveshaprabu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122986499/b555f43a-b55b-4d1a-8ec2-415f5ea985d8)


# 3. Checking the null function()
![image](https://github.com/niveshaprabu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122986499/8f8a7977-0ff6-4341-989a-37391c629b18)


# 4.Data Duplicate
![image](https://github.com/niveshaprabu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122986499/b5544191-acb0-4ab2-8827-10c7b3867883)


# 5.Print Data
![image](https://github.com/niveshaprabu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122986499/6d06f6e9-7e2a-4dd0-9111-8f26125de240)


# 6.Data Status
![image](https://github.com/niveshaprabu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122986499/c1b5ebeb-ef80-43a3-9cb8-e0c8a0e7f038)


# 7.y_prediction array
![image](https://github.com/niveshaprabu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122986499/823b260e-7095-4a86-9c10-82ac8cf46e5a)


# 8.Accuracy value
![image](https://github.com/niveshaprabu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122986499/644f0acf-e3a9-497b-8b53-a9e7c62e410d)


# 9.Confusion matrix
![image](https://github.com/niveshaprabu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122986499/232b72ff-7af9-4c70-bed0-18a945f6d03f)


# 10.Classification Report
![image](https://github.com/niveshaprabu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122986499/8ab32752-6dc1-40cc-ba2e-b02e86288d1c)


# 11.Prediction of LR
![image](https://github.com/niveshaprabu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/122986499/0dd0a038-67ee-4f25-ab8a-0bdda51b9b30)


# Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
