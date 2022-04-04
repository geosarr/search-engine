from collection import index_collection
from loadcollection import load_collection_cran, load_collection_ms_marco
from indices import InvertedIndex, PositionalIndex
from search import  term_freq, doc_freq, unigram_model
import pickle


def to_pickle(obj, name, path='./data'):
    '''
    Saving an object to pickle
    '''
    with open(path+'/'+name+'.pickle', 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        

        
def read_pickle(name, path='./data'):
    '''
    Loading a pickle object
    '''
    with open(path+'/'+name+'.pickle', 'rb') as inp:
        return pickle.load(inp)
        
        
        
def save_load(dataset="ms_marco", save=True, save_path="./data"):
    dico={"cran": load_collection_cran(), "ms_marco": load_collection_ms_marco()}
    if save:
        print(".....................Inverted Index with sorted postings.....................")
        inverted_index_sortpost=index_collection(dico[dataset], InvertedIndex(include_char_index=True, ngram=3))
#         print(".....................Inverted Index with unsorted postings.....................")
#         inverted_index_unsortpost=index_collection(dico[dataset], InvertedIndex(sort_postings=False))
        print(".....................Positional Index with sorted postings.....................")
        positional_index_sortpost=index_collection(dico[dataset],PositionalIndex(sort_postings=True))
#         print(".....................Positional Index with unsorted postings.....................")
#         positional_index_unsortpost=index_collection(dico[dataset],PositionalIndex(sort_postings=False))
        print(".....................Term and document frequencies.....................")
        term_freqs_log,term_freqs_raw,doc_freqs=term_freq(positional_index_unsortpost),term_freq(positional_index_unsortpost, 'raw'),doc_freq(positional_index_unsortpost)
        print(".....................Unigram.....................")
        unigram=unigram_model(term_freqs_raw)
        print(".....................Saving the data.....................")
        try:
            to_pickle(inverted_index_sortpost, name="inverted_index_sortpost")
            to_pickle(inverted_index_unsortpost, name="inverted_index_unsortpost")
            to_pickle(positional_index_sortpost, name='positional_index_sortpost'); 
            to_pickle(positional_index_unsortpost, name='positional_index_unsortpost');
            to_pickle(term_freqs_log, name='term_freqs_log');to_pickle(term_freqs_raw, name='term_freqs_raw');
            to_pickle(doc_freqs, name='doc_freqs');to_pickle(unigram, name="unigram")
        except:
            print(".....................Failed to save the data.....................")
    else:
        print(".....................Loading the data.....................")
        try:
            inverted_index_sortpost=read_pickle("inverted_index_sortpost");
            inverted_index_unsortpost=read_pickle("inverted_index_unsortpost")
            positional_index_sortpost=read_pickle('positional_index_sortpost')
            positional_index_unsortpost=read_pickle('positional_index_unsortpost');
            term_freqs_log=read_pickle('term_freqs_log');term_freqs_raw=read_pickle('term_freqs_raw')
            doc_freqs=read_pickle("doc_freqs"); unigram=read_pickle("unigram")
        except:
            print(".....................Failed to load the data.....................")
    
    return inverted_index_sortpost, inverted_index_unsortpost, positional_index_sortpost, positional_index_unsortpost, term_freqs_log, term_freqs_raw, doc_freqs, unigram
        
   
