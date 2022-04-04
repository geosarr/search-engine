from dataclasses import dataclass, field
import bisect
from preprocess import inverted_index_preprocessing, positional_index_preprocessing, character_ngram, clean



@dataclass
class InvertedIndex:
    index: dict=field(default_factory=dict)  # stores the index
    raw_freq: dict=field(default_factory=dict) # stores the number of occurrences of tokens in the documents they appear
    documents: dict=field(default_factory=dict) # stores the documents by ID, used when retrieving the relevant documents
    proc_terms: dict=field(default_factory=dict) # stores the processed terms of each document
    sort_postings : bool=True  # says whether or not the postings are sorted
    char_t_index: dict=field(default_factory=dict) # character to term index 
    t_char_index: dict=field(default_factory=dict) # term to character index
    include_char_index: bool=False # says whether or not to include the (term to) character (to term) index
    ngram: int=2 # the number of characters to consider for the character n-gram index
    
    def index_document(self, document):
        '''
        index the documents with or without sorted posting lists
        '''
        if document.ID not in self.documents:
            self.documents[document.ID] = document
            self.proc_terms[document.ID] = list()
        
        # Character indexing the document
        if self.include_char_index:
            cleaned_doc_terms=clean(document.content.strip()).split()
            for term in cleaned_doc_terms:
                chars=character_ngram(term, self.ngram)
                self.t_char_index[term]= chars
                for char in chars:
                    if char not in self.char_t_index:
                        self.char_t_index[char]=set()
                    self.char_t_index[char].add(term)
                
        # Invert indexing the document   
        terms = inverted_index_preprocessing(document.content) 
        self.proc_terms[document.ID]=terms
        for token in set(terms):
            if self.sort_postings:
                if token not in self.index:
                    self.index[token] = list()
                    self.raw_freq[token] = dict() 
                bisect.insort(self.index[token], document.ID) 
                # self.index[token].append(document.ID) works as well if the documents are indexed iteratively with increasing IDs.
                self.raw_freq[token][document.ID]=terms.count(token) # works if the documents are indexed iteratively one by one

                
            else:
                if token not in self.index:
                    self.index[token] = set()
                self.index[token].add(document.ID) 
         
            

                
                
@dataclass
class PositionalIndex:
    index: dict=field(default_factory=dict) 
    documents: dict=field(default_factory=dict)
    sort_postings : bool=True  
    char_t_index: dict=field(default_factory=dict) 
    t_char_index: dict=field(default_factory=dict) 
    include_char_index: bool=False 
    ngram: int=2 
            
    def index_document(self, document):
        '''
        index the documents 
        '''
        if document.ID not in self.documents:
            self.documents[document.ID] = document
        
        # Character indexing the document
        if self.include_char_index:
            cleaned_doc_terms=clean(document.content.strip()).split()
            for term in cleaned_doc_terms:
                chars=character_ngram(term, self.ngram)
                self.t_char_index[term]=chars
                for char in chars:
                    if char not in self.char_t_index:
                        self.char_t_index[char]=set()
                    self.char_t_index[char].add(term)
        
        # Positional indexing
        terms = positional_index_preprocessing(document.content)
        for position, token in enumerate(terms):
            if token not in self.index:
                # The first element is the document frequency (1 here) the following elements are the document ID followed 
                # by the raw frequency of the term and its positions  in the corresponding documents.
                self.index[token]=[1, {document.ID: [1, [position]]}]
            else:
                # The term appeared at least once (either in the current document or in a previous one)
                if self.sort_postings: IDs=[id_ for dico in self.index[token][1:] for id_ in dico] 
                else: IDs=self.index[token][1].keys()
                if document.ID in IDs:
                    # the term appears at least a second time in the current document
                    # We first increment its raw frequency
                    if self.sort_postings: 
                        pos=IDs.index(document.ID); self.index[token][pos+1][document.ID][0]+=1
                        self.index[token][pos+1][document.ID][1].append(position) 
                    else: self.index[token][1][document.ID][0]+=1; self.index[token][1][document.ID][1].append(position)
                else:
                    # The term appears in a new document
                    # We first increment its document frequency
                    self.index[token][0]+=1
                    if self.sort_postings:
                        bisect.insort(IDs, document.ID); pos=IDs.index(document.ID) 
                        self.index[token].insert(pos+1, {document.ID: [1, [position]]}) 
                    else: self.index[token][1][document.ID]=[1, [position]]
