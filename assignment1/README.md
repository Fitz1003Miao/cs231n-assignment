Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Spring 2017.

[TOC]

## KNN

kNN classifer consists of two stages:

* During training, the classifier takes the training data and simply remembers it.
* During testing, kNN classifies every test image by comparing to all training images and transfering the labels of the k most similar training examples.
* The value of k is cross-validated.

### 1. Calculate Distance

#### 1. Two loops

```python
def compute_distances_two_loops(self, x):
    num = x.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros([num, num_train])
    
    for i in range(num):
        for j in range(num_train):
            # use euclidean metric
            dists[i, j] = np.sqrt(np.sum(np.square(x[i] - self.X_train[j]))) 
 	return dists
```

#### 2. One loop

```python
def compute_distances_one_loop(self, x):
    num = x.shape[0]
    num_train = self.X_train.shapes[0]
    dists = np.zeros([num, num_train])
    
    for i in range(num):
        dists[i] = np.sqrt(np.sum(np.square(x[i] - self.X_train), axis = 1))
    return dists
```

#### 3. No loop

```python
def compute_distances_no_loops(self, x):
    num = x.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros([num, num_train])
    # dists = np.sqrt(np.sum(np.square(x[:, np.newaxis, :] - self.X_train), axis = 2)) # Cause Memory Error
    dists = np.sqrt(np.sum(np.square(x), axis = 1).reshape(-1, 1) + np.sum(np.square(self.X_train), axis = 1) - 2 * np.dot(x, self.X_train.T))
    return dists
```

> There is a small trick to compare two matrix.
>
> np.linalg.norm(a - b, ord = 'fro')

### 2. Pred labels

```python
def predict_labels(self, dists, k = 5):
    num = dists.shape[0]
    y_pred = np.zeros([num])
    for i in range(num):
        closest_y = []
        closest_y = self.y_train[np.argsort(dists[i, :])[:k]]
        from collections import Counter
        y_pred[i] = Counter(closest_y).most_common()[0][0]
       	
    return y_pred
```

## SVM

- implement a fully-vectorized **loss function** for the SVM
- implement the fully-vectorized expression for its **analytic gradient**
- **check your implementation** using numerical gradient
- use a validation set to **tune the learning rate and regularization** strength
- **optimize** the loss function with **SGD**
- **visualize** the final learned weights

### 1. Loss function

#### 1. naive

```python
def svm_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
   	for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] += -X[i]
                
    loss /= num_train
    loss += reg * np.sum(W ** 2)
    dW += 2 * reg * W
    
    return loss, dW
    
```

#### 2. vectorized

```                                                                                                                                                                                                                                                                                                                                               python
def svm_loss_vectorized(W, X, y, reg):
    num = X.shape[0]
    num_classes = W.shape[1]
    
    dW = np.zeros_like(W)
    scores = np.dot(X, W)
    correct_scores = scores[list(np.arange(num)), y].reshape(-1, 1)
    loss = (np.sum(np.maximum(scores - correct_scores + 1, 0)) - num) / num + reg * np.sum(W * W)
    
    X_mask = np.zeros_like(scores)
    X_mask[(scores - correct_scores + 1) > 0] = 1
    X_mask[list(np.arange(num)), y] -= 1
    X_mask[list(np.arange(num)), y] = -np.sum(X_mask, axis = 1)
    dW = np.dot(X.T, X_mask) / num + 2 * reg * W
    return loss, dW
```

### 2. Numerical Gradient

```python
def grad_check_sparse(f, x, analytic_grad, num_checks = 10, h = 1e-5):
    for i in xrange(num_checks):
       	ix = tuple([randrange(m) for m in x.shape])
        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval
        
        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = anaylytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        return rel_error
```



## Softmax

- loss function
- implement the fully-vectorized expression for its **analytic gradient**
- **check your implementation** with numerical gradient
- use a validation set to **tune the learning rate and regularization** strength
- **optimize** the loss function with **SGD**
- **visualize** the final learned weights

### 1. Loss function

#### 1. naive

```python
def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    
    num_train = X.shape[0]
    num_classes = W.shape[1]
  	
    for i in range(num_train):
       	scores = np.dot(X[i, :], W)
        scores -= np.max(scores)
        correct = np.exp(scores[y[i]])
        loss += -np.log(correct / np.sum(np.exp(scores)))
        
        for j in range(num_classes):
            dW[:, j] += (np.exp(scores[j])) / np.sum(np.exp(scores)) * X[i]
            if j == y[i]:
                dW[:, j] += -X[i]
                
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W ** 2)
    dW += 2 * reg * W
    
    return loss, dW
```

#### 2. vectorized

```python
def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    
    scores = np.dot(X, W)
    scores -= np.max(scores, axis = 1, keepdims = True)
    scores = np.exp(scores)
    
    loss = np.sum(-np.log(scores[list(np.arange(num_train)), y] / np.sum(scores, axis = 1))) / num_train + reg * np.sum(W ** 2)
    
    scors = scores / np.sum(scores, axis = 1, keepdims = True)
    scores[list(np.arange(num_train)), y] -= 1
    dW += np.dot(X.T, scores) / num_train + 2 * reg * W
    
    return loss, dW
```

## Features



