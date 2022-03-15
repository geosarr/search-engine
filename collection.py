from dataclasses import dataclass # practical for object oriented coding have a look here https://docs.python.org/3/library/dataclasses.html
from tqdm import tqdm

def index_collection(documents, indices):
    '''
    # INPUT:     ## documents is the collection of documents 
                 ## indices is an iterable of indices like [inverted index, positional index ]
    
    # OUTPUT:    ## loading the index to use for retrieval:
    '''
    n=1000
    for document in tqdm(documents, total=n):
        for index in indices: index.index_document(document)
    return index


@dataclass
class Collection:
    '''collection of documents characteristics'''
    size: int # the maximal number of documents in the collection

@dataclass
class Document(Collection):
    '''documents'''
    ID: int  # identifier of the document
    content: str  # content of the document

