from dataclasses import dataclass # practical for object oriented coding have a look here https://docs.python.org/3/library/dataclasses.html

def index_collection(documents, index):
    '''
    # INPUT:     ## documents is the collection of documents 
                 ## index is a type of index like inverted index, positional index
    
    # OUTPUT:    ## loading the index to use for retrieval:
    '''
    judgments=None
    for document in documents:
        # this if condition is used to get the judgments of documents wrt queries
        if type(document) in [dict]:
            judgments=document
        else: index.index_document(document)
    return index, judgments

@dataclass
class Document:
    '''documents'''
    ID: int  # identifier of the document
    content: str  # content of the document

