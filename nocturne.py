import os
import glob
import math
import sys

import bs4
import re
import MeCab
m = MeCab.Tagger("-Owakati")
files = glob.glob("dataset/nocturne/*.html")
total = len(files)
ban   = [(eg, total, name) for eg, name in enumerate(files)]
def mapper(arr):
  eg, total, name = arr

  resset = []
  with open(name) as f:
    soup = bs4.BeautifulSoup( f.read() )
    res  = soup.findAll("div", {"class": "novel_view"})
    if res == []:
      return None
    print(eg, "/", total, name, file=sys.stderr)
    r = res[0]
    try:
      text = r.text
      text = text.replace("　", "").replace("\t", "").replace(" ", "")
      text = re.sub(r"\n{1,}", "\n", text)
      for box in text.split('\n'):
        resset.append( str(m.parse(box).strip()) ) 
    except AttributeError as e:
      print(e)
      return None
  return "\n".join(resset)
      

import concurrent.futures
with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
  for res in executor.map(mapper, ban):
    print(res)
  #[mapper(b) for b in ban]
