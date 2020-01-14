## Data preparation

An application should be represented as a feature vector to be fed into machine learning models. The dataset is provided with values that should be converted to a form that machine learning algorithms can handle. 
Binary feature vector: Xϵ{0,1}^M, where Xi=1 if the feature i exists in the application and Xi=0 if the feature is not included in the application, with M=545,333.

The data needs to be processed in order to transform into a form that could be provided to machine learning algorithms. That is, representation of the variables as binary vectors, a technique called one-hot encoding. The conversion steps include mapping the char variables into integers, i.e., label encoding, and then, each integer value should be represented as a binary vector. For example, if we had to deal with only three unique features, namely, internet, getDeviceId and write_external_storage the process would be as follows:
1)	Label encoding: The values map to 0-1-2.
2)	One-hot Encoding: The integers from the previous step map to [1 0 0] – [0 1 0] – [0 0 1].


### 1) Label encoding

First, we map all application features into a numeric form of integers. We create a dictionary with all unique features present in the dataset, iterating through files line by line, with a key the feature and value an incremental integer. For example, the feature feature::android.hardware.touchscreen maps as 0. Once the entire dictionary is created, for each application, we create another file with the application’s SHA256 name, in which we append the numeric value of each feature. That is, the newly created file contains the indexes that map to original features. For example, an app is represented as {81598-0-15597-15-16-17-15598-3297-18-15599-20-114}.

```
python3 label_encoding.py
```

### 2) Grid-search

Before feeding data on ML algorithms, we have to determine the optimal hyper-parameters for each model. The goal of this operation is to construct a model with performance as high as possible. Notice that the result of this procedure will not yield the final performance of each model; instead, parameter tuning is only used to observe the best combinations in the hyper-parameters of each model. 

#### 2.1) Neural net:
Unfortunately, when tried the GridSearchCV module from Scikit-learn package [wrapping Keras](https://keras.io/scikit-learn-api/) methods for automatic grid search with a [3-fold cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation) in a set with 2000 samples, the model could not fit in our memory. In such a way, we implement parameter tuning manually. In each trial, we use the same samples to serve as training set with a total of 2000 applications and with a 0.3 malware ratio, using 3-fold cross-validation. For further details of the values of hyperparameters we recommend to read the official [Keras documentation](https://keras.io/).

In our dummy network’s architecture, we use an input layer which takes the total number of features (545,333) and feeds them to 2 hidden layers with 200 neurons in each layer and 2 neurons in the output layer as we deal with a binary classification task. As for the values that determine the learning process, we use the [ReLU](https://keras.io/activations/) activation function in hidden layers and the Softmax activation function in the output layer with a [dropout rate](https://keras.io/layers/core/) of 0.5, using the [ADAM](https://keras.io/optimizers/) optimizer. Finally, we define the [sparse categorical cross entropy](https://keras.io/losses/) to calculate the loss as our target classes are mutually exclusive integers (each sample belongs to one class and is either benign (class 0) or malicious (class 1)) and [accuracy](https://keras.io/metrics/) as the evaluation metric. In each trial we vary the value of the parameters to determine which fit the best in our task. Note that the learning rate as well as the [initialization values of weights and biases](https://keras.io/initializers/) has to be tuned. Notice that there are no hard defined rules for selecting the combination of hyper-parameters. The optimal combination is different for each dataset and the parameters require adjustments to it.

```
python3 nn_grid_search.py
```


We found that the optimal architecture is with 2 hidden layers with 200 neurons in each layer and the optimal hyperparameters using the Adam optimizer are:


•	batch size: 150 

•	epochs: 5

•	dropout rate: 0.2

•	activation function in hidden layers: ReLU

•	weights initialization: glorot_uniform

•	bias initialization: zeros

•	learning rate: 0.001

Finally, we used the same architecture, batch size, epochs, dropout rate, activation function and weights and biases initialization to train neural networks with the same architecture with other optimizers. It is shown that most optimizers perform quite well and reach Adam in terms of accuracy, even though the hyperparameters are adapted to it. The rest of the optimizers can perform better if the grid search process is repeated individually for each one.


### 2.2) Classic ml

The grid search procedure also applies to the other models considered, namely Decision Tree, Random Forest, k-Nearest Neighbors, Logistic Regression and Support Vector Machines. Note that Bayesian classifiers have only one hyperparameter for smoothing the data and we leave this parameter at its default value. 

```
python3 models_grid_search.py
```


#### [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html):

•	criterion: gini 

•	splitter: best

•	max_features: None

#### [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

•	n_estimators: 10 

•	criterion: entropy

•	max_features: log2

#### [k-Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

•	n_neighbors: 3 

•	weights: distnace

•	metric: minkowsi

#### [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

•	C: 2.0

#### [Support Vector Machines](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)

•	kernel: linear 

•	C: 0.5


### 3) Training models

In the main training procedure we train our models with 1500 applications and we evaluate them with another sample of 1500 applications. As in the process of finding the best hyperparameters, we put more effort in training neural networks. Keras supports the [early stopping](https://keras.io/callbacks/) technique. We specify the validation loss as the performance measure to be minimized. Keras seeks for the minimum validation loss as epochs go by and automatically stops the training procedure when there is no further improvement based on the ‘patience’ argument. The patience argument indicates the delay of epochs on which Keras does not determine any improvement. We define the number of epochs to 20 and the patience value to 10. This means that our models can be trained for up to 20 epochs, but if no improvement in performance is recognized before the patience value reaches, the process will stop automatically. The minimum validation loss does not guarantee that the model has the best performance in terms of accuracy on the validation set. For that reason, we define the ModelCheckpoint callback, which controls the maximum validation performance and whenever the output is increased the model is stored. 

```
python3 train_models.py
```

We implemented the aforementioned technique for each optimizer Keras supports. The results are as follows:

| Optimizer | Min. Val. loss | Epoch | Checkpoint Acc. | Epoch | Val. loss |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| RMSprop | 0.1792 | 2 | 94.67 | 5 | 0.2006 |
| Adagrad | 0.1746 | 3 | 94.67 | 4 | 0.1904 |
| Adadelta | 0.1654 | 6 | 94.66 | 15 | 0.2158 |
| Adamax | 0.1712 | 4 | 94.87 | 8 | 0.2038 |
| NAdam | 0.1853 | 3 | 94.73 | 6 | 0.2346 |
| Adam | 0.1867 | 4 | 94.47 | 4 | 0.1867 |
| SGD | 0.1632 | 15 | 94.2 | 20 | 0.1653 |
| SGD with momentum | 0.1653 | 9 | 94.33 | 15 | 0.1730 |

As for the other classifiers we do not check them for overfitting with a specific method. We implement a random subsampling technique and we evaluate the models in terms of average accuracy and standard deviation. 
We also evaluate each one of our neural network models with the value of epochs found in the previous step that achieves the highest accuracy. The training procedure on random sets is repeated 8 times, a value selected at random.

```
python3 random_subsampling.py
```

| Model | Avg. Acc. | St. Dev | 
| ------------- | ------------- |  ------------- |
| NB | 92.32 | 0.91 | 
| DT | 92.84 | 0.55 | 
| RF | 93.48 | 0.72 | 
| k-NN | 92.95 | 0.62 | 
| LR | 94.64 | 0.55 | 
| SVM | 93.9 | 0.35 | 
| RMSprop | 94.28 | 0.69 | 
| Adagrad | 94.79 | 0.67 | 
| Adadelta | 94.2 | 0.45 | 
| Adamax | 94.87 | 0.39 | 
| NAdam | 94.9 | 0.55 | 
| Adam | 94.53 | 0.84 | 
| SGD | 94.65 | 0.5 | 
| SGD with momentum | 94.5 | 0.7 | 

We observe that training in random samples does not make any significant performance difference. As such, we randomly select 1500 samples from the dataset to serve as training set and other 1500 samples to serve as testing set. Note that the neural network models trained and stored in the previous procedure with the exact same training and testing sets, when early stopping and ModelCheckpoint callback implemented. 

```
python3 train_models.py
```


| Model | Accuracy | FPR | FNR | 
| ------------- | ------------- | ------------- | ------------- | 
| NB | 92.7 | 2.8 |  18 | 
| DT | 90.8 | 7.8 |  12.4 | 
| RF | 93.33 | 3 |  15 | 
| k-NN | 93.67 | 2.95 |  14 | 
| LR | 94.13 | 4.5 |  8.8 | 
| SVM | 93.8 | 5.5 |  7.7 | 
| RMSprop | 94.67 | 4.1 |  8.2 | 
| Adagrad | 94.66 | 4.7 |  6.6 | 
| Adadelta | 94.67 | 4.1 |  8 | 
| Adamax | 94.87 | 3.7 |  8.9 | 
| NAdam | 94.73 | 4.3 |  7.3 | 
| Adam | 94.47 | 3.6 |  10 | 
| SGD | 94.2 | 4.3 |  9 | 
| SGD with momentum | 94.33 | 3 |  9.7 | 


Although, it has been shown that models have similar behavior in different sets, it is necessary to show that the trained models perform well in other random sets, other than the set of 1500 applications. This is to confirm that the models are indeed able to distinguish malware from benign application with high probability. We randomly choose 8 sets of 1500 random applications and we evaluate the performance of each trained model. 

```
python3 evaluate_models.py
```

| Model | Avg. Acc. | Avg. FPR | Avg. FNR |
| ------------- | ------------- | ------------- | ------------- |
| NB | 92.09 | 2 |  21.5 | 
| DT | 91.73 | 6 |  13.5 | 
| RF | 93.72 | 2.76 |  14.4 | 
| k-NN | 93.67 | 3 |  14 | 
| LR | 94.6 | 3.8 |  8 | 
| SVM | 93.9 | 5.2 |  7.5 | 
| RMSprop | 94.82 | 3.8 |  8.3 | 
| Adagrad | 94.88 | 4.4 |  6.6 | 
| Adadelta | 94.84 | 3.4 |  9.2 | 
| Adamax | 95.11 | 3.5 |  8.1 | 
| NAdam | 94.6 | 4.5 |  7.5 | 
| Adam | 94.53 | 3.1 |  10.8 | 
| SGD | 94.36 | 3.9 |  9.58 | 
| SGD with momentum | 95.05 | 3.2 |  8.7 | 


### 4) Incremental Learning

Instead of learning on the entire dataset at once or on a sample of a dataset (as shown in the previous evaluations), the algorithms can dynamically adapt to new samples and patterns. This is the case for Naive Bayes and neurals networks. There are any state-of-the-art implementations of incremental learning for Decision Tree, Random Forest, k-Nearest Neighbors, Logistic Regression and Support Vector Machine-based classifiers. The way the data is fed to the algorithms is such that all malware applications go through the learning algorithm exactly once. We implement this by feeding 1,000 samples per iteration, 300 of which are malicious. The total iterations are 19, the total malware samples 5,560 and the total benign samples 13,440. However, we cannot rely on the results of this technique because the total amount of malware in the wild is huge and cannot be replaced with just 5,560 malicious applications. The incremental learning procedure is mainly implemented to observe the behavior of models with the presence of an adversary.  The performance, at least in neural networks, is greatly increased, reaching about 98.5% in the same validation set of 1500 applications used in the previous evaluations. 


```
python3 incremental_learning.py
```

| Model | Accuracy | FPR | FNR |
| ------------- | ------------- | ------------- | ------------- |
| NB | 93.13 | 6.8 |  7.1 | 
| RMSprop | 97.67 | 1.1 |  5.1 | 
| Adagrad | 97.93 | 1.1 |  4.2 | 
| Adadelta | 98.13 | 0.95 |  4 | 
| Adamax | 98.13 | 1.1 |  3.6 | 
| NAdam | 98.4 | 1 |  2.8 | 
| Adam | 98.46 | 1.3 |  2 | 
| SGD | 97.4 | 1.4 |  5.3 | 
| SGD with momentum | 97.3 | 1.5 |  5.3 | 







### 5) Crafting Adversarial examples with [FGSM](https://arxiv.org/abs/1412.6572)

The attack is based on the [Tensorflow tutorial](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm).

First, we get the gradient of the loss with respect to input. Second, we produce the adversarial examples by multiplying the gradient of the loss with a noise ε and adding the result to the original input. The epsilon in computer vision domain is relatively small since the goal is to produce inputs that can fool machine learning models without affecting human decision in the sense that they do not appear to change the outlook of an image. In malware classification, epsilon can be any size since one cannot even comprehend the original data itself. However, we stick with the original methodology and we keep the value of epsilon rather low, 0.01 specifically. We evaluate the neural network models produced with a testing set of 1000 random applications, with a 0.3 malware ratio. We also evaluate the incremental learned models with the same testing set. 

```
python3 fgsm.py
```


All of the trained models with the sample of 1500 get useless since they only reach up to 5.5% accuracy (with the NAdam optimizer). SGD optimizer performs the worst with only 0.1% accuracy. As for the model which is deceived the most, the model trained with the Adadelta optimizer is the most affected, at a misclassification rate of 94.4%.  The misclassification rate is calculated based on the original accuracy of the model and its accuracy on adversarial examples. This does not lead us to a conclusion as to whether an optimizer affects the performance of a model in adversarial examples, since they all have a misclassification rate greater than 90%.

| Model | Accuracy | FPR | FNR | Adv. Accuracy | Adv. FPR | Adv. FNR | MR |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| RMSprop | 95.3 | 3.71 | 7 | 2.7 | 96.14 | 100 | 92.6 | 
| Adagrad | 95.4 | 4.1 | 5.6 | 2.5 | 96.86 | 99 | 92.9 | 
| Adadelta | 95.2 | 3.6 | 7.7 | 0.8 | 98.8 | 100 | 94.4 | 
| Adamax | 95.4 | 3.3 | 7 | 3.5 | 95 | 100 | 91.9 |
| NAdam | 95.6 | 3.9 | 5.6 | 5.5 | 93.72 | 96.33 | 90.1 |
| Adam | 95.1 | 3.3 | 8.7 | 2.5 | 96.42 | 100 | 92.6 |
| SGD | 94.3 | 4.6 | 8.3 | 0.1 | 99.9 | 100 | 94.2 | 
| SGD with momentum | 94.6 | 3.6 | 9.6 | 0.3 | 99.57 | 100 | 94.3 |

As for the incremental learned models the results are quite similar, except that they are a little more efficient in terms of accuracy. The model trained with the RMSprop optimizer performs the best on adversarial achieving 28.8% accuracy and Adagrad performs the worst, achieving 0.89% accuracy.

| Model | Accuracy | FPR | FNR | Adv. Accuracy | Adv. FPR | Adv. FNR | MR |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| RMSprop | 98.6 | 0.42 | 3.7 | 2.7 | 96.14 | 100 | 95.9 | 
| Adagrad | 98.7 | 0.57 | 3 | 0.89 | 98.71 | 100 | 97.81 | 
| Adadelta | 98.5 | 0.42 | 3.6 | 21.4 | 76.71 | 83 | 77.4 | 
| Adamax | 98.6 | 0.85 | 7 | 8.9 | 8.9 | 90 | 89.7 |
| NAdam | 98.8 | 0.85 | 5.6 | 13.5 | 80.43 | 94 | 85.3 |
| Adam | 98.8 | 0.85 | 8.7 | 8 | 8 | 91.86 | 90.8 |
| SGD | 98.3 | 4.6 | 0.85 | 6.3 | 6.3 | 98 | 92 | 
| SGD with momentum | 98.3 | 4 | 9.6 | 7.1 | 91.14 | 97 | 91.2 |

The FGSM attack is entirely based on the gradients of the loss function, a non-realistic scenario for malware classification task, since it randomly adds a noise to the input domain. With this method the most likely is that the applications that will be created to be non-functional (though capable of deceiving machine learning models). This is not what an adversary wants, since the purpose is not only the deception, but to maintain the functionality of an application as well. However, an adversary to whom the parameters of a model are known can create a copy of the malicious application and whenever the application passes though the detector, the adversary can push the features created by the FGSM attack. Thereby, it can overtake a detector. This is a difficult accomplishment as an adversary has to push these features somehow. Therefore, the creation of a new application based on the original malware with the property of bypassing a detector is a more realistic scenario.

### 6) Crafting Adversarial examples with [JSMA](https://arxiv.org/pdf/1511.07528.pdf)

The attack is based on a [JSMA variant](https://arxiv.org/pdf/1606.04435.pdf).

The steps are similar to the original proposition of JSMA with the difference that we deal with a binary classification task. As such, in the first step the attack makes use of the forward derivative of the model F, based on the predicted class F(x), the input dimension m (=545,333 features) and the output dimension of 2 classes.The steps are similar to the original proposition of JSMA with the difference that we deal with a binary classification task. As such, in the first step, the attack makes use of the forward derivative of the model F, based on the predicted class F(x), the input dimension m (=545,333 features) and the output dimension of 2 classes.The calculation of the forward derivative is made upon in each input data, evaluating the model’s outcome at each input separately. In essence, the forward derivative estimates the direction in which a perturbation would change the model’s outcome by first computing the gradient of the model with respect to the input. The second step is to find the small change δ in x with the maximal positive gradient into the target class y'=0.  In other words, we compute an index i that maximizes the change into the target class by changing the value of x at the index i. The limitation with this method is that changes occur only when the maximal index value is 0. This means that we only add features, without removing any, changing the value of x_i from 0 to 1. With this method we ensure that an application remains functional and able to bypass security mechanisms, at least our malware detector. In essence, we obtain a new feature vector for the sample x. This is only applies for one change in the input dimension, but such a small change (by changing only one value) cannot ensure that can cause the model to misclassify a sample. Therefore, we need more changes until misclassification is achieved. To achieve this, we re-compute the gradient under the new input vector and we find another feature to change. The process is repeated until the misclassification is reached or the amount changes reached the limitation for the maximum amount of changes we allow. The value of the maximum changes is the second limitation in this methodology. We set the maximum changes to k = 20 as the work done in  to ensure that the algorithm cannot produce more than 20 perturbations in the original feature vector.   

```
python3 jsma.py
```


First, we evaluate the models trained with the sample of 1500 applications. As observed, with the JSMA variant the most robust model is that with the Adagrad optimizer, which achieves 71.2% overall performance and 86.3% false negative ratio with 12.3 average changes. 

| Model | Accuracy | FNR | Adv. Accuracy | Adv. FNR | MR | Distortion |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| RMSprop | 95.3 | 7 | 68.5 | 96.3 | 89.3 | 7.34 | 
| Adagrad | 95.4 | 5.6 | 71.2 | 86.3 | 80.7 | 12.3 | 
| Adadelta | 95.2 | 7.6 | 68.5 | 97 | 89.4 | 7.54 |
| Adamax | 95.4 | 7 | 68.5 | 97.3 | 90.3 | 8.92 |
| NAdam | 95.6 | 5.6 | 68.6 | 96 | 90.4 | 8.89 |
| Adam | 95.1 | 8.7 | 71.3 | 88 | 79.3 | 11.33 |
| SGD | 94.3 | 8.3 | 68.3 | 95 | 86.7 | 7.8 | 
| SGD with momentum | 94.6 | 9.6 | 68.5 | 97 | 87.4 | 6.7 |

Second, we evaluate the incremental learned models. The most robust model is with the SGD with momentum optimizer, which achieves 71% overall performance and 95% false negative ratio with only 6.9 average changes. 


| Model | Accuracy | FNR | Adv. Accuracy | Adv. FNR | MR | Distortion |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| RMSprop | 98.6 | 3.7 | 69.7 | 100 | 96.3 | 5.71 | 
| Adagrad | 98.7 | 3 | 70.6 | 97 | 94 | 7.24 | 
| Adadelta | 98.5 | 3.7 | 70.4 | 97.3 | 93.6 | 5.8 |
| Adamax | 98.6 | 3.7 | 70.1 | 98.6 | 94.9 | 6.61 |
| NAdam | 98.8 | 2 | 69.4 | 100 | 98 | 5.95 |
| Adam | 98.8 | 2 | 69.6 | 99.3 | 97.3 | 7.21 |
| SGD | 98.3| 3.7 | 70.6 | 96 | 92.3 | 6.3 | 
| SGD with momentum | 98.3 | 4 | 71 | 95 | 91 | 6.9 |


### 7) Defense with [adversarial training](https://adversarial-ml-tutorial.org/adversarial_training/).

Steps for our model:
1)	Train a classifier F.
2)	Craft adversarial examples A for F using the JSMA crafting method. 
3)	Use additional training epochs on F with the A produced in the previous step as training data that corresponds to the malicious class. 

We adapt the methodology of adversarial training in the incremental learned model with Adam optimizer trained with the whole feature space. Note that with the ModelCheckpoint callback we found that 4 epochs per iteration produce the model with the highest accuracy with Adam optimizer (at least in the training procedure on the whole feature space). Therefore, we iterate 4 additional epochs to train the models with adversarial examples. There is any specific methodology to follow for adversarial training such as how many perturbations needed or whether to mix adversarial examples with legitimate input data. We iterate 4 additional epochs on our trained model with adversarial examples with a 1.0, 0.7, 0.5 and 0.3 ratio, correspondingly. The rest of training set is filled with benign applications. In each trial the adversarial examples are produced from a random set of malicious applications. 

```
python3 adversarial_train.py
```


The best overall performance on adversarial examples for the original model achieved with the additional training of 800 samples (240 adversarial and 560 legitimate benign samples) with a 0.3 adversarial examples ratio (Accuracy:98.4, FPR:1.42, FNR:2). 

However, to determine which model performs the best, the evaluation in adversarial examples crafted for another model is not sufficient. Therefore, we evaluate the performance of the produced models to random sets without the presence of an adversary. It is determined that the 0.3 ratio in adversarial examples with 800 samples in total, produces the best model in terms of performance both with and without the presence of an adversary as it  achieves 98.52% average accuracy in 8 random validation sets. As for the validation set of 1500 used to evaluate the original model, the results are quite similar. In the model trained with 0.3 ratio in adversarial examples, we can observe a slight decrease in average performance (from 98.46% to 97.73%) with an increase in the FPR (from 1.3% to 2.48%), while the FNR is slightly decreased (from 2% to 1.78%). Consequently, with the additional training with 800 samples of which 240 are adversarial examples we get a model with performance as high as the original one. 

An adversary powerful enough to fool the original model may be able to deceive the adversarial trained models. In previous evaluations we did not consider the robustness of the trained networks to adversarial examples specifically crafted for these models, but only their resistance to adversarial examples for the original model. Therefore, we craft adversarial examples for the additional trained models with the JSMA variant to evaluate their robustness against adversarial examples. The results show that adversarial trained models are harder to attack than the original model as the average amount of perturbations increases, but the misclassification rates remains quite high. The most robust model seems to be the one trained with 800 samples with a 0.5 adversarial examples ratio as it achieves about 61% false negative ratio with the presence of an adversary, requiring 15.24 average changes.  

It is easily observed that the training with adversarial examples produced for the original model does not improve the resistance of the new models against adversarial examples crafted for them, although they are making it harder for the adversary to find the perturbations. The goal of adversarial training is to produce a model that can generalize better not only for adversarial examples crafted for the original model, but for the adversarial trained model as well. Thus, we apply the adversarial training procedure continuously to the newly generated networks. We apply this method to the network trained with 0.3 adversarial examples ratio as it achieves higher performance rates in sets without the presence of an adversary. Our goal is to produce a model with as high accuracy as the original model in legitimate samples, while achieving high performance rates in classifying adversarial examples, requiring as many perturbations as possible. To implement this, we compute new adversarial examples for the newly generated classifier F'. The samples are fed back to re-train F' and generate a new network. We apply this method continuously until we get high distortion. After 12 iterations we observe the most robust model, which requires 17.21 average changes to be deceived. With the iterative adversarial training we observe that the model not only became more robust against adversarial examples, but it also has a similar behavior with the original model without the presence of adversary. The FPR is slightly affected (from 0.85% to 1.42%), but we can observe a decrease in FNR (from 2% to 1%). As such, adversarial training made the model more powerful in terms of detecting malicious applications. However, an adversary is still able to bypass the detector, although more perturbations are required. The misclassification rate fell from 97.3% to 29.33, but remains high enough.  


### 8) Defense with [Distillation](https://arxiv.org/pdf/1511.04508.pdf)

The code is based on [this](https://github.com/carlini/nn_robust_attacks/blob/master/train_models.py) script.

Steps for our model:
1)	Train a classifier F (teacher network) using standard methods except that the final classification is not given by the usual softmax activation function, but by the softmax with the addition of a temperature (T) variable. Typically, the T is a large value and as T→∞ the distribution approaches uniform. We evaluate different values of T in our experiments.
2)	The trained classifier F from the previous step is evaluated on each sample of the training set. As a result it creates two new labels that concern the probability of a sample to belong to either benign or malicious class (soft labels). In other words, instead of generating an output that reflects the model’s belief that a sample belongs to a class, soft labels contain the probability that a sample belongs to one or to the other class. For example, an application has a 70% change of being in malicious class and a 30% chance of being in benign class. Without the T value, the classifier should predict that the sample is a malicious application. Instead, with a large T the classifier outputs the probabilities that correspond to classes.
3)	Train a second classifier F' (student/distilled network) using the temperature T on the soft labels produced by the classifier F. 
4)	The classifier F' is evaluated on each sample with T=1. 

We consider only the implementation of defensive distillation with the Adam optimizer with the training sample of 1500. We do not implement the defense mechanism with incremental learning due to the high computation cost. Recall that in the main training procedure (without the presence of adversaries) we used the sparse categorical cross entropy as the function to measure the loss since we had to deal with mutual exclusive integers (0 for the benign class or 1 for the malicious one). Defensive distillation requires the model’s belief for a sample to belong in a class. To this end, we use the categorical cross entropy based on logits to measure the loss. The logits means that the function operates based on the output of the last hidden layer. In other words, with the use of logits the final output values may not be equal to 1. Moreover, the use of the categorical cross entropy requires the labels to be in one-hot encoding form. This change in the loss function is expected to slightly affect the accuracy of the model, but without a high diverge. As such, we make use of the EarlyStopping and ModelCheckpoint callback to determine the value of epochs that produces the most accurate model.

```
python3 defensive_distillation.py
```

We found that the optimal epoch value is 7 in this case. Indeed, the performance is not significantly affected, as the model achieves a slightly higher performance than the original model (94.6 from the original 94.47) on the validation set of 1500 applications. In defensive distillation, the temperature parameter is that which affects the final resilience of the model in adversarial examples. As such, we experiment with different temperatures, varying its value from 10 to 200. More specifically we measure the misclassification rate for the following temperatures: {10, 20, 40, 70, 100, 120, 150, 200}. As observed, defensive distillation can defend against adversarial examples as the temperature value rises. However, the misclassification rate remains quite high. The lowest misclassification rate is found at temperature 150, which is 8%. The overall performance with the presence of adversarial examples is 91.5% with 20% FNR. This FNR in the malware classification domain is quite high, as the goal is to effectively detect malicious applications. Overall, with the defensive distillation we observe an increase of the FNR without the presence of an adversary as we increase the temperature parameter. On the other hand, the FPR is not affected, as we observe a slight reduction as the temperature increases. The misclassification rate drops significantly to 8% with a temperature value of 150. However, the average changes in the feature space that are able to produce misclassification are rather small (2.29 with a temperature of 150).


### 9) Ensemble Classifier
By combining multiple classifiers we hope to improve the overall performance as well as to reduce the model’s misclassification rate on adversarial examples. We combine classifiers in such a way that each one contributes equally to the final prediction. As a result, the final output is a majority voting approach. For the final prediction, each model in the ensemble, individually evaluate a sample and output its prediction. We evaluate the ensemble approach with a small test set of 200 applications, 100 of which are malicious. We use both deep models and classic machine learning algorithms to form the ensemble. To craft adversarial examples for the ensemble, we use the JSMA variant based on the deep model trained with the Adam optimizer. In other words, we assume an adversary that is unaware of the ensemble and craft adversarial examples only for the model trained with the Adam optimizer substitute model of a deep model based on the Adam optimizer. This model has an original accuracy of 98% (with 2% FNR) and 49% (with 100% FNR) on adversarial examples.


```
python3 ensemble.py
```

All of the tested configurations perform quite similar as they achieve higher than 97% and up to 98.8% accuracy. Indeed, the overall performance is slightly better than the performance using a model alone. Moreover, the ensemble learning does improve the resistance on adversarial examples. The highest misclassification rate is 25% by using only deep learning models. Combining deep models that calculates the loss and gradients quite differently (e.g. Adam and SGD) with classic machine learning algorithms produces robust enough models. The overall optimal ensemble is found by combining two deep learning models with the logistic regression algorithm, achieving 98.5% original accuracy with 8% misclassification rate on adversarial examples. Nonetheless, a tuning strategy has to implement to determine the best ensemble approach. 

### Helper scripts


#### set_one_hot_encoding.py

perform the one hot encoding technique automatically.

#### models.py

contains the configuration for classic ml algorithms as well as methods for training, testing and evaluating them.

#### neural_network.py

contains the configuration for the neural network as well as methods for training, testing and evaluating it.
