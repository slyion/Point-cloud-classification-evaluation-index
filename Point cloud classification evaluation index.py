
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score   # 准确率
from sklearn.metrics import precision_score  # 精确率/查准率
from sklearn.metrics import recall_score       # recall/查全率
from sklearn.metrics import f1_score           # F1
from sklearn.metrics import mean_absolute_error # numpy MAE（Mean Absolute Error：平均绝对误差）
from sklearn.metrics import mean_squared_error  # numpy MSE（Mean Square Error：均方误差）
from sklearn.metrics import confusion_matrix   #  混淆矩阵

# https://blog.csdn.net/u011630575/article/details/79645814?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170460307616800211557571%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170460307616800211557571&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~top_positive~default-1-79645814-null-null.nonecase&utm_term=accuracy_score&spm=1018.2226.3001.4450

# sklearn.metrics中的评估方法介绍
# (accuracy_score, recall_score, roc_curve, roc_auc_score, confusion_matrix)


classkind = [0,1,2]
kkk =   [2, 0, 2, 2, 0, 1, 1, 1,0,0,0,0,0,0,0,0]  # 预测标签列表
pred_squ=np.array(kkk)
jjj = [0, 0, 2, 2, 0, 2, 1, 0,1,1,1,1,1,1,1,1]  # 真实标签列表
target_squ=np.array(jjj)

end_acc = accuracy_score(target_squ, pred_squ)  # 计算准确率
end_preci2 = precision_score(target_squ, pred_squ, average='macro')  # 计算精确率
end_recall2 = recall_score(target_squ, pred_squ, average='macro')  # 计算召回率
end_F1 = f1_score(target_squ, pred_squ, average='weighted')  # 计算F1分数
end_mae = mean_absolute_error(target_squ, pred_squ)  # 计算平均绝对误差
end_mse = mean_squared_error(target_squ, pred_squ)  # 计算均方误差
endmat = confusion_matrix(target_squ, pred_squ)  # 生成混淆矩阵数据
# 绘制混淆矩阵的热力图
sns.heatmap(endmat, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')  # 设置x轴标签
plt.ylabel('True labels')  # 设置y轴标签
plt.title('Confusion Matrix')  # 设置图表标题
plt.show()  # 展示图表
#


#print("miou", sum(arr_iou_boll) / len(arr_iou_boll)) # miou计算
print("acc", end_acc)
print("preci", end_preci2)
print("recall", end_recall2)
print("F1", end_F1)
print("mae",end_mae)
print("mse", end_mse)
print("endmat\n", endmat)        # 画出混淆矩阵图


#ious计算
segp = pred_squ
segl = target_squ
part_ious = [0.0 for _ in range(len(classkind))]
for l in classkind:
    if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
        part_ious[l - classkind[0]] = 1.0
    else:
        part_ious[l - classkind[0]] = np.sum((segl == l) & (segp == l)) / float(
            np.sum((segl == l) | (segp == l)))

####  part_ious 是一个 list ##############
print(part_ious[0])
print(part_ious[1])
print(part_ious[2])








