import xlrd as xd
import numpy as np
import math as mh
from matplotlib import pyplot as plt
from matplotlib import pylab as plb

#读取数据
data = xd.open_workbook('data.xls')
table = data.sheets()[0] 

#数据初始化
sex_men = []
sex_women = []
height_men = []
height_women = []
weight_men = []
weight_women = []
sex = table.col_values(1)[1:]
height = table.col_values(3)[1:]
weight = table.col_values(4)[1:]

#对数据中男女进行分类
for index, value in enumerate(sex):
    if value == 1:
        sex_men.append(int(index))
        height_men.append(height[index])
        weight_men.append(weight[index])
    else:
        sex_women.append(int(index))
        height_women.append(height[index])
        weight_women.append(weight[index])

#数据可视化（直方图）
plt.figure('Height Distribution')       
plb.subplot(1, 2, 1) 
plt.hist(height_men,len(set(height_men)))
plt.title('Height Histogram(men)')
plt.xlabel('height(cm)')
plt.ylabel('number')
plb.subplot(1, 2, 2)
plt.hist(height_women,len(set(height_women)))
plt.title('Height Histogram(women)')
plt.xlabel('height(cm)')
plt.ylabel('number')
plt.savefig('height histogram.jpg')
plt.figure('Weight Distribution') 
plb.subplot(1, 2, 1) 
plt.hist(weight_men,len(set(weight_men)))
plt.title('Weight Histogram(men)')
plt.xlabel('weight(kg)')
plt.ylabel('number')
plb.subplot(1, 2, 2)
plt.hist(weight_women,len(set(weight_women)))
plt.title('Weight Histogram(women)')
plt.xlabel('weight(kg)')
plt.ylabel('number')
plt.savefig('weight histogram.jpg')   
    
#采用最大似然估计方法，求男女生身高以及体重分布的参数
#样本数据的均值
height_men_average = sum(height_men) / len(sex_men)
height_women_average = sum(height_women) / len(sex_women)
weight_men_average = sum(weight_men) / len(sex_men)
weight_women_average = sum(weight_women) / len(sex_women)

#样本数据的方差
sigma_height_men = sum([(x - height_men_average) ** 2 for x in height_men]) / len(sex_men)
sigma_weight_men = sum([(x - weight_men_average) ** 2 for x in weight_men]) / len(sex_men)
sigma_height_women = sum([(x - height_women_average) ** 2 for x in height_women]) / len(sex_women)
sigma_weight_women = sum([(x - weight_women_average) ** 2 for x in weight_women]) / len(sex_women)

#输出结果
print('\n'+'-'*25+'最大似然估计'+'-'*25+'\n')
print('男生\n')
print('身高均值: %f,    方差: %f\n'%(height_men_average,sigma_height_men))
print('体重均值: %f,    方差: %f\n'%(weight_men_average,sigma_weight_men))
print('女生\n')
print('身高均值: %f,    方差: %f\n'%(height_women_average,sigma_height_women))
print('体重均值: %f,    方差: %f\n'%(weight_women_average,sigma_weight_women))

#采用贝叶斯估计方法，求男女生身高以及体重分布的参数
#假定男女生身高体重的方差的先验值为10
sigma_height_menx = 10;      
sigma_weight_menx = 10;  
sigma_height_womenx = 10;  
sigma_weight_womenx = 10;

#贝叶斯估计
uN1=(sigma_height_men * sum(height_men) + sigma_height_menx * height_men_average) / (sigma_height_men * len(sex_men) + sigma_height_menx);  
uN2=(sigma_weight_men * sum(weight_men) + sigma_weight_menx * weight_men_average) / (sigma_weight_men * len(sex_men) + sigma_weight_menx);  
uN3=(sigma_height_women * sum(height_women) + sigma_height_womenx * height_women_average) / (sigma_height_women * len(sex_women) + sigma_height_womenx);  
uN4=(sigma_weight_women * sum(weight_women) + sigma_weight_womenx * weight_women_average) / (sigma_weight_women * len(sex_women) + sigma_weight_womenx);

#输出结果
print('\n'+'-'*20+'最小错误率贝叶斯估计'+'-'*20+'\n');
print('假定男女生身高和体重先验分布的方差均为10,估计结果为：\n');
print('男生\n') 
print('身高均值: %f，    体重均值: %f\n' % (uN1, uN2))
print('女生\n')
print('身高均值: %f，    体重均值: %f\n' % (uN3, uN4)) 

#采用最小错误率贝叶斯决策，画出类别判定的决策面。
#求协方差矩阵
c1 = [x - height_men_average for x in height_men]
c2 = [x - weight_men_average for x in weight_men]
C12 = sum([x * y for x, y in zip(c1, c2)])        
c3 = [x - height_women_average for x in height_women]
c4 = [x - weight_women_average for x in weight_women]
C34 = sum([x * y for x, y in zip(c3, c4)])
sigma12 = C12 / len(sex_men)
sigma34 = C34 / len(sex_women)
sigma_men = np.matrix([[sigma_height_men,sigma12], [sigma12,sigma_weight_men]])     
sigma_women = np.matrix([[sigma_height_women,sigma34],[sigma34,sigma_weight_women]])   

#求先验概率
pre_men = len(sex_men) / len(sex)
pre_women = len(sex_women) / len(sex)

#决策面函数
def g_function(x1, x2):
    N_1 = np.matrix([x1 - height_men_average, x2 - weight_men_average])  
    N_2 = np.matrix([x1 - height_women_average, x2 - weight_women_average])
    g = 0.5 * N_1 * (sigma_men.I) * N_1.T - 0.5 * N_2 * (sigma_women.I) * N_2.T + 0.5 * mh.log(np.linalg.det(sigma_men) / np.linalg.det(sigma_women)) - mh.log(pre_men / pre_women)
    return g

#判断(160,45)和(178,70)时应该属于男生还是女生
#决策面可视化
sample_x1 = 160
sample_y1 = 45
sample_x2 = 178
sample_y2 = 70

x1 = np.arange(130, 200, 1)
x2 = np.arange(30, 100, 1)
x1, x2 = np.meshgrid(x1, x2)
g = np.zeros(x1.shape)

for i in range(x1.shape[0]):
    for j in range(x1.shape[1]):
        g[i][j] = g_function(x1[i][j], x2[i][j]) 
        
plt.figure()
plt.contour(x1, x2, g, 0)
plt.scatter(height_men,weight_men, s=70, c='r', marker='x')
plt.scatter(height_women,weight_women, s=70, c='b', marker='o')
plt.scatter(sample_x2, sample_y2, s=200, c='g', marker='*')
plt.scatter(sample_x1, sample_y1, s=200, c='y', marker='*')
plt.title('Bias decision surface')
plt.xlabel('height(cm)')
plt.ylabel('weight(kg)')
plt.savefig('Bias decision.jpg')
plt.show()

