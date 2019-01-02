Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Spring 2017.
## knn
knn 练习的两个重点是:
* 距离函数的实现
* 如何选出类别

### 距离计算
1. 两次循环计算
```Python
for i in xrange(num_test):
      for j in xrange(num_train):
          dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
```
2. 一次循环计算
```Python
for i in xrange(num_test):
    dists[i] = np.sqrt(np.sum(np.square(X[i] - self.X_train), axis = 1))
```

3. 直接使用 NumPy 计算
```Python
X_square = np.square(X)
X_train_square = np.square(X_train)
X_mul_X_train = np.dot(X, X_train.T)
dists = np.sqrt(np.sum(X_square, axis = 1).reshape(-1, 1) + np.sum(X_train_square, axis = 1) - 2 * X_mul_X_train)
```

### 选出类别
```Python
for i in xrange(num_test):
    closest_y = []
    closest_y = self.y_train[np.argsort(dists[i])[:k]]
    from collections import Counter
    y_pred[i] = Counter(closest_y).most_common()[0][0]
```

## SVM
* 损失函数
* 梯度

$$ loss = max(0, s_j - s_{y_i}) + margin $$

通常 margin 为1

1. 循环计算
```Python
for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:, y[i]] += -X[i]

loss /= num_train
dW /= num_train
loss += reg * np.sum(W * W)
dW += 2 * reg * W
```

2. NumPy 计算
```Python
scores = X.dot(W)
correct_scores = scores[list(np.arange(num_train)), y]
scores = scores - correct_scores + 1
loss = np.sum()
```

