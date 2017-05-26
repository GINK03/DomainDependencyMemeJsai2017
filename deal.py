import os
import glob
import math
import sys
import re
import MeCab
import pickle
import sklearn
import numpy as np
from sklearn.cluster import KMeans
""" YahooNewを分かち書きしてストアする """
def step1():
  m = MeCab.Tagger("-Owakati")
  names  = glob.glob('../YahooNewsScraper/output/*/*')
  alllen = len(names)
  for en, name in enumerate(names):
    if en%500 == 0:
      print(en, alllen, name)
    with open(name, 'r') as f, open("yahoo.news.txt", "a") as w:
      for line in f:
        line  = line.strip()
        lines = filter(lambda x:x!="", re.split(r"(\n)", line) )
        for e in lines:
          wakati = m.parse(e).strip()
          w.write( wakati + "\n" )
          #print(wakati + "\n")

def step2():
  """ まずベクトルを作る、この時に特に問題のデータを取り出す """
  os.system("./fasttext skipgram -input yahoo.news.txt -output model  -thread 16 -maxn 0")

""" ここで文章のベクトル化を行って、出力する"""
def step3():
  key_vec = {}
  maxx    = 11080000
  for i in range(0, maxx, 10000):
    print(i, maxx)
    res = os.popen("head -n {i} ./dataset/yahoo.news.txt | tail -n 10000 | ./fasttext print-sentence-vectors ./models/model.bin".format(i=i)).read()
    for line in res.split("\n"):
      if line == "":
        continue
      vec = list(map(float, line.split()[-100:]))
      txt = line.split()[:-100]
      key = " ".join(txt)
      if key_vec.get(key) is None:
        key_vec[key] = vec
  open("key_vec.pkl", "wb").write(pickle.dumps(key_vec))

""" ベクトルをダンプして、教師なしクラスタリングをする """
def step4():
  key_vec = pickle.loads(open("key_vec.pkl", "rb").read()) 
  vecs = []
  for ev, vec in enumerate(key_vec.values()):
    x = np.array(vec)
    if np.isnan(x).any():
      # print(vec)
      continue
    vecs.append(x)
  vecs   = np.array(vecs)
  kmeans = KMeans(n_clusters=128, init='k-means++', n_init=10, max_iter=500,
                       tol=0.0001,precompute_distances='auto', verbose=0,
                       random_state=None, copy_x=True, n_jobs=1)
  kmeans.fit(vecs)
  
  open("kmeans.model", "wb").write( pickle.dumps(kmeans) )
  for p in kmeans.predict(vecs):
    print(p)

""" 文脈skipgramを構築する """
import concurrent.futures
import json
def _step5(arr):
  kmeans = pickle.loads(open("kmeans.model", "rb").read())
  key, lines = arr
  print(key)
  open("./tmp/tmp.{key}.txt".format(key=key), "w").write("\n".join(lines))
  res  = os.popen("./fasttext print-sentence-vectors ./models/model.bin < tmp/tmp.{key}.txt".format(key=key)).read()
  objs = []
  for line in res.split("\n"):
    #print(line)
    try:
      vec = list(map(float, res.split()[-100:]))
    except:
      print(line)
      print(res)
      #sys.exit()
      continue
    x = np.array(vec)
    if np.isnan(x).any():
      continue
    cluster = kmeans.predict([vec])
    txt = res.split()[:-100]
    obj = {"txt": txt, "cluster": cluster.tolist()} 
    #print(obj)
    objs.append(obj)
  open("tmp/tmp.{key}.json", "w").write(json.dumps(objs))
  
def step5():
  key_lines = {}
  with open("dataset/yahoo.news.txt", "r") as f:
    for el, line in enumerate(f):
      line = line.strip()
      key  = el//1000
      if key_lines.get(key) is None:
        key_lines[key] = []
      key_lines[key].append(line)
  #with concurrent.futures.ProcessPoolExecutor() as executor:
  key_lines = [(k,l) for k,l in key_lines.items()]
  for k_l in key_lines:
    _step5(k_l)
      

if __name__ == '__main__':

  if '--step1' in sys.argv:
    step1()

  if '--step2' in sys.argv:
    step2()
  
  if '--step3' in sys.argv:
    step3()

  if '--step4' in sys.argv:
    step4()
  
  if '--step5' in sys.argv:
    step5()
