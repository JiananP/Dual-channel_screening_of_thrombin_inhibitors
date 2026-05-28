import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel("RFE_selected_features_30.xlsx")
X = data.iloc[:, :-1]  # 特征变量
y = data.iloc[:, -1]   # 目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建支持向量机模型
svm_model = SVC()

# 定义参数网格
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf']}  # 使用RBF内核

# 使用网格搜索调整参数
grid_search = GridSearchCV(SVC(), param_grid, cv=10)
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 重新拟合最佳模型
best_model.fit(X_train, y_train)

# 预测
y_pred = best_model.predict(X_test)

# 十折交叉验证
cv_scores = cross_val_score(best_model, X_train, y_train, cv=10)


# 计算各项性能指标
accuracy = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = recall_score(y_test, y_pred)
specificity = tn / (tn + fp)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# 计算AUC值
y_probs = best_model.decision_function(X_test)
auc_score = roc_auc_score(y_test, y_probs)

# 输出性能指标
print("Test Set Performance:")
print("Accuracy:", accuracy)
print("Sensitivity (Recall):", sensitivity)
print("Specificity (SP):", specificity)
print("Precision:", precision)
print("F1 Score:", f1)
print("MCC:", mcc)
print("AUC Score:", auc_score)

# 绘制AUC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()