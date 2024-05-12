# EXNO:02 Exploratory Data Analysis
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("/content/titanic_dataset.csv")
df
```
![328032655-fb53c826-c322-4048-aa0e-5be94f93b46b](https://github.com/Swetha733N/EXNO2DS/assets/122199934/8187a330-c2db-40e4-ba38-138867b380a5)


```
df.info()
```
![328032669-556fcaa2-66be-43cd-be60-4c39ab0bebb8](https://github.com/Swetha733N/EXNO2DS/assets/122199934/00f5f0c8-119b-458c-b4bf-bb2a024551f2)


```
df.set_index("PassengerId",inplace =True)
df.shape
```
![328032683-3e4e3dba-4e45-40e0-a9db-711c6d58aca1](https://github.com/Swetha733N/EXNO2DS/assets/122199934/48a262b1-11ef-4dcf-b1a3-7ea91e9bb87e)


```
df.nunique()
```
![328032697-94d5229b-21d2-497d-a85c-863ffd1a787f](https://github.com/Swetha733N/EXNO2DS/assets/122199934/33a48b84-996d-464a-9684-af9790ec8c96)


```
df["Survived"].value_counts()
```
![328032710-28af9088-03cd-4840-8055-70642d83323c](https://github.com/Swetha733N/EXNO2DS/assets/122199934/d7840d00-8dc9-420f-8929-8bc56fc9d4d1)

```
per=(df['Survived'].value_counts()/df.shape[0]*100).round(2)
per
```
![328032724-a0a3f07d-8110-46b3-ad3f-3a4a54ea4e22](https://github.com/Swetha733N/EXNO2DS/assets/122199934/0ce7632e-ce69-434c-bdcb-528fd101293c)


```
sns.countplot(data=df,x="Survived")
```
![328032737-e2a87a26-8915-4ef3-8c1d-8b5d89a34560](https://github.com/Swetha733N/EXNO2DS/assets/122199934/2c2374ee-90d5-407b-aaa9-eed0763ec347)


```
fig, ax1 = plt.subplots(figsize=(5,5))
graph=sns.countplot(ax=ax1,x= 'Survived', data=df)
graph.set_xticklabels (graph.get_xticklabels (), rotation=90)
for p in graph.patches:
  height = p.get_height()
  graph.text(p.get_x()+p.get_width()/2., height + 0.1, height,ha="center")
```
![328032753-bb472590-a60b-4c21-a3b6-0ce1e5f3ad5c](https://github.com/Swetha733N/EXNO2DS/assets/122199934/a6f69169-7233-438f-b5ef-acba9042f314)


```
df.Pclass.unique()
```
![328032771-e17da48a-01de-4931-a084-6f5c5e46774a](https://github.com/Swetha733N/EXNO2DS/assets/122199934/8bdeee55-6246-4857-ae04-66a9abd22a23)


```
df.rename(columns = {'Sex':"Gender"},inplace=True)
df
```

```
sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=.7)
```
![328032784-8aeef7e3-f7e7-40c6-bc93-bbf55d87de33](https://github.com/Swetha733N/EXNO2DS/assets/122199934/912dd846-a8ab-4ded-8237-bbb05cac7ec2)


```
sns.catplot(x="Survived",hue="Gender",data=df,kind="count")
```
![328032802-aaabd4a3-fbac-45ff-a06c-f64429f85c05](https://github.com/Swetha733N/EXNO2DS/assets/122199934/809288d2-5f81-4e58-8fe5-b845e05a2498)


```
fig, ax1 = plt.subplots(figsize=(8,5))
graph=sns.countplot(ax=ax1,data=df,x="Survived", hue="Pclass", palette="rainbow")
graph.set_xticklabels (graph.get_xticklabels())
for p in graph.patches:
  height = p.get_height()
  graph.text(p.get_x()+p.get_width()/2, height+ 20.8, height,ha="left")
```
![328032811-df255f97-6f55-4058-b682-c78badb02382](https://github.com/Swetha733N/EXNO2DS/assets/122199934/81613ff7-35f2-419e-a9d6-07c00a1b6899)


```
df.boxplot(column="Age",by="Survived")
```
![328032824-30150408-f147-4bbb-bc8f-27a83892036f](https://github.com/Swetha733N/EXNO2DS/assets/122199934/52404bfd-8572-4e15-bf5e-6b1f6c955097)

```
sns.scatterplot(x=df["Age"],y=df["Fare"])
```
![328032854-0b80db12-3099-4383-823d-8d7b5ec5b86d](https://github.com/Swetha733N/EXNO2DS/assets/122199934/32b46ef9-3268-46e9-9e7f-b66a59b8b28f)


```
sns.jointplot(x="Age",y="Fare",data=df)
```
![328032867-07541157-1db3-44a8-828d-454e52439238](https://github.com/Swetha733N/EXNO2DS/assets/122199934/1786cfc2-7f4c-4bfd-9cd9-e5738f703a16)


```
fig,ax1=plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue="Gender",data=df)
```
![328032882-306e6aec-811f-4e27-b8ec-3cd000da0bac](https://github.com/Swetha733N/EXNO2DS/assets/122199934/78d46bbd-0dfe-4d23-a290-fec284db7908)


```
sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![328032892-a16384e3-ffbe-4201-8a68-dd9a4cc77e4b](https://github.com/Swetha733N/EXNO2DS/assets/122199934/b39bb410-ed36-4edc-a11c-1757af2c6b82)


```
g= sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass", kind = "count", legend=True)
g.fig.set_size_inches(8,5)
g.fig.subplots_adjust(top=0.81,right=0.86)
ax =g.facet_axis(0,0)
for p in ax.patches:
ax.text(p.get_x()-0.01,p.get_height()*1.02,'{0:.1f}'.format(p.get_height()),color='red',rotation='horizontal',size='small')
```
![328032902-9b8372e2-e6d8-43e6-9ed1-9c8770e02cbd](https://github.com/Swetha733N/EXNO2DS/assets/122199934/246be342-3cb1-4b9d-8574-231ae4db3d24)


```
corr=df.corr()
sns.heatmap(corr,annot=True)
```
![328032915-5ed76665-d728-42ff-8ea4-4f5636b0cb02](https://github.com/Swetha733N/EXNO2DS/assets/122199934/183444c1-0b53-4402-b41c-de1be309c3de)


```
sns.pairplot(df)
```

![328032926-31bbc5fd-c168-430b-aba0-f8b028d21c45](https://github.com/Swetha733N/EXNO2DS/assets/122199934/40f18a8b-f9de-4e72-8c4b-126dfe8fd4b1)



# RESULT
Thus, the outputs verifies that the data set has been applied the EDA process and methods.
