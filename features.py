import ipaddress as ip  # works only in python 3
from os.path import splitext
import pandas as pd
from urllib.parse import urlparse  # 对url进行解析
import tldextract  # 用于获取域名和后缀

# 2016's top most suspicious TLD and words
Suspicious_TLD = ['zip', 'cricket', 'link', 'work', 'party', 'gq', 'kim', 'country', 'science', 'tk']
Suspicious_Domain = ['luckytime.co.kr', 'mattfoll.eu.interia.pl', 'trafficholder.com', 'dl.baixaki.com.br',
                     'bembed.redtube.comr', 'tags.expo9.exponential.com', 'deepspacer.com', 'funad.co.kr',
                     'trafficconverter.biz']


class GetFeatures:

    def countdots(self, url):
        """统计.出现的次数"""
        return url.count(".")

    def countdelim(self, url):
        """统计以下标点符号出现的次数"""
        count = 0
        delim = [';', '_', '?', '=', '&']
        for each in url:
            if each in delim:
                count = count + 1

        return count

    def isip(self,  url):
        """url中是否出现ip地址"""
        try:
            if ip.ip_address(url):
                return 1
        except:
            return 0

    def isPresentHyphen(self, url):
        """-出现的次数"""
        return url.count('-')

    def isPresentAt(self, url):
        """@出现的次数"""
        return url.count('@')

    def isPresentDSlash(self, url):
        """//出现的次数"""
        return url.count('//')

    def countSubDir(self, url):
        """/出现的次数"""
        return url.count('/')

    def get_ext(self, url):
        """分离文件名和扩展名"""
        root, ext = splitext(url)
        return ext

    def countSubDomain(self, subdomain):
        """子域名的个数"""
        if not subdomain:
            return 0
        else:
            return len(subdomain.split('.'))

    def countQueries(self, query):
        if not query:
            return 0
        else:
            return len(query.split('&'))


    def get_features(self, url, label):
        result = []
        url = str(url)

        # add the url to feature set
        result.append(url)

        # parse the URL and extract the domain information
        path = urlparse(url)
        ext = tldextract.extract(url)

        # counting number of dots in subdomain
        result.append(self.countdots(ext.subdomain))

        # checking hyphen in domain
        result.append(self.isPresentHyphen(path.netloc))

        # length of URL
        result.append(len(url))

        # checking @ in the url
        result.append(self.isPresentAt(path.netloc))

        # checking presence of double slash
        result.append(self.isPresentDSlash(path.path))

        # Count number of subdir
        result.append(self.countSubDir(path.path))

        # number of sub domain
        result.append(self.countSubDomain(ext.subdomain))

        # length of domain name
        result.append(len(path.netloc))

        # count number of queries
        result.append(len(path.query))

        # Adding domain information

        # if IP address is being used as a URL
        result.append(self.isip(ext.domain))

        # presence of Suspicious_TLD
        result.append(1 if ext.suffix in Suspicious_TLD else 0)

        # presence of suspicious domain
        result.append(1 if '.'.join(ext[1:]) in Suspicious_Domain else 0)


        # result.append(get_ext(path.path))
        result.append(str(label))
        return result
