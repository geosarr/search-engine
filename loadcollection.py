from collection import Document
import re

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