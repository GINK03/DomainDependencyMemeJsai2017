import os
import glob
import math
import sys
import re
import MeCab
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

if __name__ == '__main__':
  if '--step1' in sys.argv:
    step1()
