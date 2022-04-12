import pandas as pd
from features import GetFeatures
from tqdm import tqdm
import sklearn.ensemble as ek
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression


def save_feats():
    data = pd.read_csv('数据/small_dataset/urls.csv')
    # data = data.copy().sample(frac=1).reset_index(drop=True)
    # print(data.head())
    getFeatures = GetFeatures()
    featureSet = pd.DataFrame(columns=('url', 'no of dots', 'presence of hyphen', 'len of url', 'presence of at', \
                                       'presence of double slash', 'no of subdir', 'no of subdomain', 'len of domain',
                                       'no of queries', 'is IP', 'presence of Suspicious_TLD', \
                                       'presence of suspicious domain', 'label'))
    for i in tqdm(range(len(data))):
        features = getFeatures.get_features(data['url'].loc[i], data['label'].loc[i])
        featureSet.loc[i] = features
    print(featureSet)
    featureSet.to_csv('数据/small_dataset/feats.csv')


def main():
    featureSet = pd.read_csv('数据/small_dataset/feats.csv', index_col=0)
    X = featureSet.drop(['url', 'label'], axis=1).values
    y = featureSet['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    model = {"DecisionTree": tree.DecisionTreeClassifier(max_depth=10),
             "RandomForest": ek.RandomForestClassifier(n_estimators=50),
             "Adaboost": ek.AdaBoostClassifier(n_estimators=50),
             "GradientBoosting": ek.GradientBoostingClassifier(n_estimators=50),
             "GNB": GaussianNB(),
             "LogisticRegression": LogisticRegression()
             }
    results = {}
    for algo in model:
        clf = model[algo]
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("%s : %s " % (algo, score))
        results[algo] = score

    winner = max(results, key=results.get)
    print(winner)
    clf = model[winner]
    res = clf.predict(X)
    mt = confusion_matrix(y, res)
    print(mt)

    result = pd.DataFrame(columns=('url', 'no of dots', 'presence of hyphen', 'len of url', 'presence of at', \
                                   'presence of double slash', 'no of subdir', 'no of subdomain', 'len of domain',
                                   'no of queries', 'is IP', 'presence of Suspicious_TLD', \
                                   'presence of suspicious domain', 'label'))

    getFeatures = GetFeatures()
    inputs = [
        ('http://www.baidu.com', "0"),
        ('https://blog.csdn.net/', "0"),
    ]
    for url, label in inputs:
        results = getFeatures.get_features(url, label)
        result.loc[0] = results
        tmp = result.drop(['url', 'label'], axis=1).values
        print(tmp)
        print(clf.predict(tmp))


if __name__ == '__main__':
    main()
    # save_feats()