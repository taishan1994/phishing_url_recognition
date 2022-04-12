# phishing_url_recognition
恶意域名识别

# 说明

数据是从[https://github.com/ebubekirbbr/phishing_url_detection](https://github.com/ebubekirbbr/phishing_url_detection)下载来的，由于数据量太大，将small_dataset下的val.txt作为训练集，并重新对训练集进行划分为训练集和测试集。识别的方法采用两种方式，一种是基于定义的特征，利用机器学习的方法进行识别，另一种是将url视为每个字符构成的序列，使用深度学习的方法进行识别。<br>

- process.py里面get_csv2()用于获取url及label
- features.py里面是机器学习相关的特征
- models.py里面是深度学习的相关模型

# 依赖

```shell
python==3.6
scikit-learn
numpy
pandas
tensorflow==1.12.0
keras==2.2.4
```

## 基于机器学习的方法

主要文件是main.py

```python
DecisionTree : 0.7682157200229489 
RandomForest : 0.7686460126219162 
Adaboost : 0.7638171734557276 
GradientBoosting : 0.7696022183973992 
GNB : 0.6845955249569707
```

# 基于深度学习的方法

主要文件是main2.py，这里只训练展示两种模型。如果还需要尝试其他模型，需要在main2.py中的fit()里面增加配置：

```python
        if self.params['architecture'] == "cnn":
            model = self.models.cnn_base()
        elif self.params['architecture'] == "rnn":
            model = self.models.rnn_base()
        else:
            raise Exception("请确认输入的模型名")
```

结果：

```
cnn_base：
 			  precision    recall  f1-score   support

  legitimate       0.92      0.86      0.89     11541
    phishing       0.84      0.90      0.87      9375

    accuracy                           0.88     20916
   macro avg       0.88      0.88      0.88     20916
weighted avg       0.88      0.88      0.88     20916

Test loss: 0.27886661878049795  |  test accuracy: 0.8793268361837555

rnn_bae：
              precision    recall  f1-score   support

  legitimate       0.87      0.93      0.90     11541
    phishing       0.90      0.83      0.87      9375

    accuracy                           0.89     20916
   macro avg       0.89      0.88      0.88     20916
weighted avg       0.89      0.89      0.89     20916

Test loss: 0.2645691215741199  |  test accuracy: 0.8860202790400932
```

# 参考

> https://github.com/ebubekirbbr/phishing_url_detection
>
> [surajr/URL-Classification: Machine learning to classify Malicious (Spam)/Benign URL's (github.com)](https://github.com/surajr/URL-Classification)

