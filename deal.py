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
  os.system("./fasttext skipgram -input dataset/bind.txt -output models/model  -thread 16 -maxn 0")

""" ここで文章のベクトル化を行って、出力する"""
def step3():
  key_vec = {}
  maxx    = 12505807
  size    = 10000
  for i in range(size, maxx, size):
    print(i, maxx)
    res = os.popen("head -n {i} ./dataset/bind.txt | tail -n {size} | ./fasttext print-sentence-vectors ./models/model.bin".format(i=i, size=size)).read()
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
  kmeans = KMeans(n_clusters=128, init='k-means++', n_init=10, max_iter=300,
                       tol=0.0001,precompute_distances='auto', verbose=0,
                       random_state=None, copy_x=True, n_jobs=1)
  print("now fitting...")
  kmeans.fit(vecs)
  
  open("kmeans.model", "wb").write( pickle.dumps(kmeans) )
  for p in kmeans.predict(vecs):
    print(p)

""" 文脈skipgramを構築する """
import concurrent.futures
import json
def _step5(arr):
  kmeans = pickle.loads(open("kmeans.model", "rb").read())
  key, lines, tipe = arr
  print(key)
  open("./tmp/tmp.{tipe}.{key}.txt".format(tipe=tipe,key=key), "w").write("\n".join(lines))
  res  = os.popen("./fasttext print-sentence-vectors ./models/model.bin < tmp/tmp.{tipe}.{key}.txt".format(tipe=tipe, key=key)).read()
  w    = open("tmp/tmp.{tipe}.{key}.json".format(tipe=tipe,key=key), "w")
  for line in res.split("\n"):
    try:
      vec = list(map(float, line.split()[-100:]))
    except:
      print(line)
      print(res)
      continue
    x = np.array(vec)
    if np.isnan(x).any():
      continue
    cluster = kmeans.predict([vec])
    txt = line.split()[:-100]
    obj = {"txt": txt, "cluster": cluster.tolist()} 
    data = json.dumps(obj, ensure_ascii=False)
    w.write( data + "\n" )
  
def step5():
  key_lines = {}
  with open("dataset/yahoo.news.txt", "r") as f:
    for el, line in enumerate(f):
      line = line.strip()
      key  = el//10000
      if key_lines.get(key) is None:
        key_lines[key] = []
      key_lines[key].append(line)
  key_lines = [(k,l,"news") for k,l in key_lines.items()]
  with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    executor.map(_step5, key_lines)
  
  key_lines = {}
  with open("dataset/nocturne.txt", "r") as f:
    for el, line in enumerate(f):
      line = line.strip()
      key  = el//10000
      if key_lines.get(key) is None:
        key_lines[key] = []
      key_lines[key].append(line)
  key_lines = [(k,l,"nocturne") for k,l in key_lines.items()]
  with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    executor.map(_step5, key_lines)

""" Windowsはば3で文脈skipを行う"""
def _step6(arr):
  name = arr
  return term_clus

def step6():

  for tipe in ["news", "nocturne"]:
    names = [name for name in reversed(sorted(glob.glob("./tmp/tmp.{tipe}.*.json".format(tipe=tipe))))]
    size  = len(names)
    for en, name in enumerate(names):
      term_clus = {}
      oss = []
      with open(name) as f:
        for line in f:
          line = line.strip()
          oss.append(json.loads(line))
      for i in range(3, len(oss) - 3):
        terms = set( oss[i]["txt"] )
        for term in terms:
          if term_clus.get(term) is None:
             term_clus[term] = [0.0]*128
          cd = [oss[i+d]["cluster"][0] for d in [-3, -2, -1, 1, 2, 3]]
          for c in cd: 
            term_clus[term][c] += 1.0
      print("{}/{} finished {}".format(en, size, name))
    open("{tipe}.term_clus.pkl".format(tipe=tipe), "wb").write( pickle.dumps(term_clus) )

def step7():
  term_clus = pickle.loads(open("./news.term_clus.pkl", "rb").read())
  term_clus = {term:clus for term, clus in filter(lambda x: sum(x[1]) > 30, term_clus.items()) }
  for term in term_clus.keys():
    vec = term_clus[term] 
    acc = sum(vec)
    term_clus[term] = list(map(lambda x:x/acc, vec))
  open("news.term_dist.pkl", "wb").write(pickle.dumps(term_clus))

  term_clus = pickle.loads(open("./nocturne.term_clus.pkl", "rb").read())
  term_clus = {term:clus for term, clus in filter(lambda x: sum(x[1]) > 30, term_clus.items()) }
  for term in term_clus.keys():
    vec = term_clus[term] 
    acc = sum(vec)
    term_clus[term] = list(map(lambda x:x/acc, vec))
  open("nocturne.term_dist.pkl", "wb").write(pickle.dumps(term_clus))

def step8():
  term_dist = pickle.loads( open("nocturne.term_clus.pkl", "rb").read() )
  for term, dist in term_dist.items():
    print(term, dist)

  ...
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

  if '--step6' in sys.argv:
    step6()

  if '--step7' in sys.argv:
    step7()

  if '--step8' in sys.argv:
    step8()
