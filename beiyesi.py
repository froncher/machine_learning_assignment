import pandas as pd
import numpy as np
Train_data = pd.read_csv(r'D:\pythonProject1\1\used_car_train_20200313.csv',sep=' ')
Test_data = pd.read_csv(r'D:\pythonProject1\1\used_car_testB_20200421.csv',sep=' ')
print('Train_data_shape: ',Train_data.shape)
print('TestB_data_shape: ',Test_data.shape)
Train_data.head()._append(Train_data.tail())#查看训练集首五行+末五行
Test_data.head()._append(Train_data.tail())#查看测试集首五行+末五行
Train_data.describe()#训练集概况
Test_data.describe()#测试集概况
Train_data.info()#查看数据缺失情况
#易知 bodyType、fuelType、gearbox缺失值较多
# 判断训练集每列的nan情况
Train_data.isnull().sum()
Test_data.info()#查看数据缺失情况
#同样是 bodyType、fuelType、gearbox缺失值较多
Test_data.isnull().sum()
#训试集
Train_data_miss=Train_data.isnull().sum()
Train_data_miss=Train_data_miss[Train_data_miss > 0]
Train_data_miss.sort_values(inplace=True)
#Train_data_miss.plot.bar()
#测试集
Test_data_miss=Test_data.isnull().sum()
Test_data_miss=Test_data_miss[Test_data_miss > 0]
Test_data_miss.sort_values(inplace=True)
#Test_data_miss.plot.bar()
df=pd.DataFrame({'Train':Train_data_miss,'Test':Test_data_miss})
ax=df.plot.bar(rot=0)


import missingno as msno
msno.matrix(Train_data.sample(500))# 训练集可视化缺失值
msno.matrix(Test_data.sample(500))# 测试集可视化缺失值
Train_data.info()
#可以发现除了notRepairedDamage为object类型其他都为int或float 继续对其进行进一步检测
Train_data['notRepairedDamage'].value_counts()#统计notRepairedDamage项数据
#推测“-”为缺失值，更换为NaN
Train_data['notRepairedDamage'].replace('-',np.nan,inplace=True)
Train_data['notRepairedDamage'].value_counts()
Test_data.info()
#测试集同上操作
Test_data['notRepairedDamage'].value_counts()#统计notRepairedDamage项数据
Test_data['notRepairedDamage'].replace('-',np.nan,inplace=True)
Test_data['notRepairedDamage'].value_counts()
import seaborn as sns #seaborn是可视化库，是对matplotlib进行二次封装而成
import matplotlib.pyplot as plt #Pyplot 是 matplotlib 的子库，用于绘制 2D 图表
Nu_feature = list(Test_data.select_dtypes(exclude=['object']).columns)  # 数值变量
Ca_feature = list(Test_data.select_dtypes(include=['object']).columns)
plt.figure(figsize=(30,25))
i=1
for col in Nu_feature:
    ax=plt.subplot(6,5,i)
    ax=sns.kdeplot(Train_data[col],color='red')
    ax=sns.kdeplot(Test_data[col],color='cyan')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax=ax.legend(['train','test'])
    i+=1
plt.show()
Train_data_features = Train_data.drop(['price'], axis=1)
# 找出所有的数值型变量
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in Train_data_features.columns:
    if Train_data_features[i].dtype in numeric_dtypes:
        numeric.append(i)

# 对所有的数值型变量绘制箱体图
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=Train_data_features[numeric], orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)
# 找出明显偏态的数值型变量
#skew_features = Train_data_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)
skew_features = Train_data_features[numeric].skew(axis=0,skipna=True).sort_values(ascending=False)
skew_features.head(31)
