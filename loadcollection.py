from collection import Document
import re
from datasets import load_dataset
from tqdm import tqdm

def load_collection_cran(judgments=None, version=None, split=None):
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
       

def load_collection_ms_marco(judgments, version, split):
    data=load_dataset('ms_marco', version, split=split)

    n=len(data)
    doc_id=1
    judgmts=dict()
    k=0
    for d in tqdm(data, total=n):
        query=d["query"]
        judgmts[query]=dict()
        for pos, text in enumerate(d["passages"]["passage_text"]):
            yield Document(ID=doc_id, content=text)
            if judgments is not None :
                # the document that is judged as relevant (its relevance is 1+is_selected)
                try: judgmts[query][doc_id]=1+d["passages"]['is_selected'][pos]
                except: pass
            doc_id+=1
        d.clear()
    yield judgmts

