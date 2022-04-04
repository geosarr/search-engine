from collection import Document
import re
import sys
import os
from lxml import etree
from gensim.corpora import WikiCorpus
import gzip

def load_collection_cran():
    with open('data/cran.all.1400', 'r') as f:
        docs=f.read().replace("\n", " ")
    f.close()
    # Preprocessing the documents here
    # but a more thorough preprocessing step will be conducted in the preprocesse.py file
    docs=re.sub(r"[0-9]", '', docs)
    docs=docs.split(".I")[1:]
    for i in range(len(docs)):
        yield Document(ID=i+1, content = docs[i].split(' .W')[1].strip())
    docs.clear() # to save memory
       

def load_collection_ms_marco():
  data=load_dataset('ms_marco', 'v1.1')
  data=data['train']
  doc_id=1
  for d in data['train']:
    for text in d["passages"]["passage_text"]:
      yield Document(ID=doc_id, content=text)
      doc_id+=1
    d.clear()
    
def load_collection_wiki(file_name='enwiki-latest-abstract.xml.gz', data_path = './data/wiki_data', size_max=1000000):
    # get the wikipedia abstract files
    file_path=data_path+"/"+file_name
    if not os.path.isfile(file_path):
        print("Downloading the data fom wikipedia dumps ...")
        os.system('wget  https://dumps.wikimedia.org/enwiki/latest/'+file_name)
        os.system('mv '+file_name+' '+data_path)
        os.system('rm -r '+file_path)
        print("Finished downloading")
    # open a filehandle to the gzipped Wikipedia dump
    with gzip.open(file_path, 'rb') as f:
        doc_id = 1
        # iterparse will yield the entire `doc` element once it finds the
        # closing `</doc>` tag
        for _, element in etree.iterparse(f, events=('end',), tag='doc'):
            content = ' '.join([element.findtext('./title'), element.findtext('./abstract')])
            
            yield Document(ID=doc_id, content=content)
            if (doc_id > size_max):
                break
                
            doc_id += 1
            
            element.clear()


    
    
