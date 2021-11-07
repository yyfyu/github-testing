# 分类
1. 导言： 监督学习任务是回归（用于预测某个值）和分类（预测某个类别）。在第二章我们探索了一个回归任务：预测房价。我们使用了多种算法，诸如线性回归，决策树，和随机森林（这个将会在后面的章节更详细地讨论）。现在我们将我们的注意力转到分类任务上。
2. mnist是机器学习 分类的入门数据集。MNIST 有 70000 张图片，每张图片有 784 个特征。这是因为每个图片都是28*28像素的，并且每个像素的值介于 0~255 之间。让我们看一看数据集的某一个数字。你只需要将某个实例的特征向量，reshape为28*28的数组，然后使用 Matplotlib 的imshow函数展示出来

## Data
* data结构：
* 用Matplotlib的imshow函数展示图
* 分训练集和测试集： 
        i. MNIST 数据集已经事先被分成了一个训练集（前 60000 张图片）和一个测试集（最后 10000 张图片）
        ii. 用np.random.permutation打乱训练集属性和label的index的顺序。这可以保证交叉验证的每一折都是相似（你不会期待某一折缺少某类数字)。一些学习算法对训练样例的顺序敏感，当它们在一行当中得到许多相似的样例，这些算法将会表现得非常差。打乱数据集将保证这种情况不会发生。:
            shuffle_index = np.random.permutation(60000)
            X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
        

## 训练一个二分类器：是5 or 非5
* 随机 stochastic梯度下降分类器 SGD：这个分类器有一个好处是能够高效地处理非常大的数据集。这部分原因在于 SGD 一次只处理一条数据，这也使得 SGD 适合在线学习（online learning）
        from sklearn.linear_model import SGDClassifier
        sgd_clf = SGDClassifier(random_state=42)
        sgd_clf.fit(X_train, y_train_5)
        sgd_clf.predict([some_digit])----True是5

## 对分类器性能的评估：     
**注意**：      
1. 对分类器的评估主要使用了sklearn.metrics包.     
2. 对线性回归的mse的评估也用了sklearn.metrics包。     
3. 对线性和分类器的交叉验证（mse和accuracy）的评估， 用sklearn.model_selection包。     
4. 交叉验证只是一种分几折的方法。如果直接用metrics包则不能分成几折。
5. 交叉验证可以用分几折的方法：计算选择model的score（cross_val_score),或者直接得到模型的预测值（cross_model_predict,本章在混淆矩阵里用了），比不分几折更精确。 
5. 所以分类器的评估标准是精确率（precision）和召回率（recall）。但因为精确率和召回率呈反比，需要trade off。所以有了根据精确率和召回率变形的F1 score，ROC， AUC来做评估。 混淆矩阵是精确率和召回率怎么来的。
6. ROC AUC PR图都是通过调整模型的决策函数，得到每一行样本的分数。把这些分数分别设为阈值，大于这个阈值的样本，分类器判断为true。小于的为false。每调试一个阈值，就得到一对FPR，TPR（ROC的横纵坐标）。所以构成了ROC图。通过ROC图的评判标准，来看这个分类器性能是不是好，是不是有overfitting。并且可以对比三个模型的ROC曲线，根据AUC面积大小，判断模型优劣。 而不是选择阈值，所以ROC图上没有阈值的值。
7. PR曲线（对精确率和召回率的权衡）可以选择最佳阈值。步骤：先算出p和r和threshold。再画出f1图vs thresholds。 选f1最大的阈值。
      
      ![Screen%20Shot%202021-11-04%20at%206.56.42%20PM.png](attachment:Screen%20Shot%202021-11-04%20at%206.56.42%20PM.png)

**检验分类模型好坏的方法**：在判断用roc还是pr判断前，要先看正反比例差距是否很大。     

    i.❤️交叉验证（不好）：
        ¥ cross_val_score: 返回的是评估分数
        ¥ 精度很高，但不代表这个分类器好的原因：当其中一些类比其他类频繁得多，精度不是好的性能度量指标。
         
    ii.❤️混淆矩阵的TPR, FPR, FNR, TNR（好）：
        ¥ cross_val_predict和confusion_matrix()： 返回的是每一个测试折，做出的预测值。
        ¥ 定义： 输出类别 A 被分类成类别 B 的次数。举个例子，为了知道分类器将 5 误分为 3 的次数，你需要查看混淆矩阵的第五行第三列。
        ¥ 步骤： 
        
            * 1.得到一些预测值，用cross_val_predict()函数，返回基于每一个测试折, 基于SGD模型, 基于X_train, 基于y_train_5做出的预测值y_train_pred。
                 ☺️from sklearn.model_selection import cross_val_predict
                   y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)☺️ 
                
            * 2.使用 confusion_matrix()函数，你将会得到一个混淆矩阵。传递目标类(y_train_5)和预测类（y_train_pred）给它。
               ☺️ from sklearn.metrics import confusion_matrix
                  confusion_matrix(y_train_5, y_train_pred)☺️
                 
            * 3. 混淆矩阵的解读 
            
     iii: ❤️准确率，精确率，召回率， F1 score： 
         ☺️from sklearn.metrics import precision_score, recall_score
           precision_score(y_train_5, y_pred)
           recall_score(y_train_5, y_train_pred)☺️
            * 准确率和召回率的折中 
        
     iiii:❤️ ROC曲线， AUC曲线， PR曲线



## 多类分类器：
* 二分类器： SVM分类器， 线性分类器， 逻辑分类器
* 多类分类器： 随机森林分类器， 朴素贝叶斯贝叶器
* 二分类器 配合 策略（ova/ova）制作多类分类器： 
    1. ova：有n个类别（0到9），有n个分类器，每个分类器是ture/false（是不是数字0）。十个分类器得到十个score，哪个score高，就是哪个数。score是这张照片等于这个分类的概率。
    2. ovo：有n个类别，有n*（n-1）/2个分类器。原理：一个分类器用来处理数字 0 和数字 1，一个用来处理数字 0 和数字 2，一个用来处理数字 1 和 2，以此类推。这叫做“一对一”（OvO）策略。如果有 N 个类。你需要训练N*(N-1)/2个分类器。 。45个分类器得到十个score，哪个score高，就是哪个数。score是这张照片等于这个分类的概率。
    3. 用ova还是ovo：大数据集用ova，小数据集用ovo。大部分二分类ova更好，所以sklearn自带的策略是ova， 除了svm是ovo。 但是可以用OneVSOneClaasifier强制改成ovo。
    4. 怎么分是多分类还是二分类，关键是看fit(X_trian, y_train)里的y_trian有几个类别。如y_trian有10个类别（数字0到数字9）。y_train_5有两个类别（True/False）
    5. 随机森林不用ova/ovo。因为自动多分类
    6. 二分类和多分类的score/proba查询：
        i. 二分类查score:用decision_function(某行），输出的是各**类别**的score，排序是根据数据里类别的排序。需要用sgd_clf.classes_[5]查询第5个类别的名字
        ii. 多分类查probability： 用forest_clf.predict_proba([some_digit])
    7. 分类器之后一定要用cross_val_score查询精度

        

#### 误差分析：用图表达混淆矩阵，与混淆矩阵的变形

* 查混淆矩阵：你需要使用cross_val_predict()做出预测，然后调用confusion_matrix()函数，


## 多标签分类器：一个样例分配到多个类别
* KNN classfier
* 比方说，这个分类器被训练成识别三个人脸，Alice，Bob，Charlie；然后当它被输入一张含有 Alice 和 Bob 的图片，它应该输出[1, 0, 1]（意思是：Alice 是，Bob 不是，Charlie 是）。这种输出多个二值标签的分类系统被叫做多标签分类系统。
* 所以先把图片变成n个位数的array。
* 带入mnist的例子：每张图有两个标签，是否是数字5，是否是奇数。[False
, True]. 

* 多标签分类器的评估：
        i: 给每一个类得到一个评估数，求平均值
               y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
               f1_score(y_train, y_train_knn_pred, average="macro")  
        ii. 这里假设所有标签有着同等的重要性，但可能不是这样。特别是，如果你的 Alice 的照片比 Bob 或者 Charlie 更多的时候，也许你想让分类器在 Alice 的照片上具有更大的权重。一个简单的选项是：给每一个标签的权重等于它的支持度（比如，那个标签的样例的数目）。为了做到这点，简单地在上面代码中设置average="weighted"。

## 多输出分类器： 
           
      
      
  
## cross_val_score/predict：都是用一个值评估我们的模型
* cross_val_score: 用accuracy/mse等评估
* cross_val_predict: 用与在得到confusion_matrix和f1 score之前，得到一些predict的值。