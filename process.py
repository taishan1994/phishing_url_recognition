import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

def get_csv():
    with open('数据/badqueries.txt', 'r', encoding='utf-8') as fp:
        bad = fp.read().strip().split("\n")
    with open('数据/goodqueries.txt', 'r', encoding='utf-8') as fp:
        good = fp.read().strip().split("\n")
    bad2 = pd.read_excel('数据/中文恶意网页列表+URL.xlsx')['网址'].tolist()
    print("bad1数量：", len(bad))  # 48126
    print("bad2数量：", len(bad2))  # 521
    print("good数量：", len(good))  # 1294531
    # 随机从good中选取50000条数据
    index = random.sample(list(range(len(good))), 50000)
    good = [good[i] for i in index]
    columns = ['url', 'label']
    res = []

    for b in bad:
        res.append((b.strip(), 1))
    for b in bad2:
        res.append((b.strip(), 1))
    for g in good:
        res.append((g.strip(), 0))
    res = pd.DataFrame(res, columns=columns)
    print(res)
    res.to_csv('数据/utl.csv')

def plot_length(featureSet):
    """绘制url长度和标签的关系"""
    sns.set(style="darkgrid")
    sns.distplot(featureSet[featureSet['label'] == 0]['len of url'], color='green', label='Benign URLs')
    sns.distplot(featureSet[featureSet['label'] == 1]['len of url'], color='red', label='Phishing URLs')
    plt.title('Url Length Distribution')
    plt.legend(loc='upper right')
    plt.xlabel('Length of URL')
    plt.show()

def plot_dots(featureSet):
    """绘制dots和标签的关系"""
    x = featureSet[featureSet['label'] == 0]['no of dots']
    y = featureSet[featureSet['label'] == 1]['no of dots']
    plt.hist(x, bins=8, alpha=0.9, label='Benign URLs', color='blue')
    # sns.distplot(x,bins=8,color='blue',label='Benign URLs')
    plt.hist(y, bins=10, alpha=0.6, label='Malicious URLs', color='red')
    # sns.distplot(y,bins=8,color='red',label='Malicious URLs')
    plt.legend(loc='upper right')
    plt.xlabel('Number of Dots')
    plt.title('Distribution of Number of Dots in URL')
    plt.show()

def plot_domain(featureSet):
    """获取域名长度和标签的关系"""
    sns.set(style="darkgrid")
    sns.distplot(featureSet[featureSet['label'] == 0]['len of domain'], color='blue', label='Benign URLs')
    sns.distplot(featureSet[featureSet['label'] == 1]['len of domain'], color='red', label='Malicious URLs')
    plt.title('Domain Length Distribution')
    plt.legend(loc='upper right')
    plt.xlabel('Length of Domain/Host')
    plt.show()


def get_csv2():
    with open('数据/small_dataset/val.txt', 'r') as fp:
        data = fp.read().strip().split('\n')
    columns = ['url', 'label']
    res = []
    for d in data:
        d = d.split('\t')
        if d[0] == 'legitimate':
            label = 0
        else:
            label = 1
        url = d[1]
        res.append((url, label))
    res = pd.DataFrame(res, columns=columns)
    res.to_csv('数据/small_dataset/urls.csv')


def get_vocab():
    with open('数据/small_dataset/val.txt', 'r') as fp:
        data = fp.read().strip().split('\n')
    res = ['PAD']
    for d in data:
        d = d.split('\t')
        url = d[1]
        for i in url:
            if i not in res:
                res.append(i)
    print("总共有：", len(res))
    with open('数据/small_dataset/vocab.txt', 'w') as fp:
        fp.write("\n".join(res))

if __name__ == '__main__':
    # get_csv()
    # data = pd.read_csv('数据/feats.csv')
    # print(data.columns)
    # data.drop(['Unnamed: 0'], axis=1, inplace=True)
    # plot_length(data)
    # plot_dots(data)
    # plot_domain(data)
    # get_csv2()
    get_vocab()