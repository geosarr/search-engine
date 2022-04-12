from collection import index_collection
from loadcollection import load_collection_cran, load_collection_ms_marco
from indices import InvertedIndex, PositionalIndex
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
        
        
        
def indexing(dataset="ms_marco", save=True, save_load_path="./data", index_type="inv", judgments=None, version=None, split=None):
    data={"cran": load_collection_cran, "ms_marco": load_collection_ms_marco}
    idx={"inv": InvertedIndex, "pos": PositionalIndex}

    if index_type not in idx:
        raise ValueError (f"the function supports only values {list(idx.keys())} for index_type")

    elif dataset not in data:
        raise ValueError ("The only supported values for argument dataset are {}".format(list(data.keys())))

    elif dataset!="ms_marco" and (version is not None or split is not None):
        raise ValueError ("Set the arguments split and version to None if dataset='ms_marco' is not used")

    if save:
        print(".....................Collection indexation in progress.....................")
        index, judgments=index_collection(data[dataset](judgments=judgments, version=version, split=split), \
                        idx[index_type](include_char_index=True, ngram=3))
        try:
            to_pickle(index, "inv", save_load_path)
            print(".....................Successfully saved the index.....................")
        except:
            print(".....................Failed to save the index.....................")
    else:
        print(".....................Loading the data.....................")
        try:
            index=read_pickle(index_type, save_load_path)
        except:
            print(".....................Failed to load the data.....................")
    
    return index, judgments
        
   
