from dataclasses import dataclass # practical for object oriented coding have a look here https://docs.python.org/3/library/dataclasses.html
#from tqdm import tqdm

def index_collection(documents, index):
    '''
    # INPUT:     ## documents is the collection of documents 
                 ## index is the type of index (inverted index or positional index used)
    
    # OUTPUT:    ## loading the index to use for retrieval:
    '''
    for document in documents:
        index.index_document(document)
        if document.ID%200==0:
            print('{} document(s) have been indexed'.format(document.ID))
    return index


@dataclass
class Document:
    '''documents'''
    ID: int
    content: str

