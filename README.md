This is an ongoing project on a search engine on textual documents.

# Vocabulary
In the following, you will some vocabulary used to comment the codes:

- document: designates basically a text

- collection: is a set of documents

- token/term: is an instance of a text/document after breaking it down and preprocessing it.

- document frequency of a token: is the number of times the token appears in a collection

- raw frequency/frequency of a token: is the number of times the token appears in a given document

- index: designates a kind of tokens collection of information relative to the documents. For instance the inverted index of a collection can be seen as a dictionary in which keys are tokens and values the identifiants of the documents in which they appear, like the index seen on a book as explained here https://bart.degoe.de/building-a-full-text-search-engine-150-lines-of-code/. A positional index is an extension of an inverted index, the main difference being that the position of the token in the documents in which it appears is added, along with the token's document frequency and raw frequencies. For example, in an inverted index like {'coding': [1,36], 'in': [1, 15, 490], 'python': [1], ...}, the token 'coding' appears in documents 1 and 36, the token 'in' appears in documents 1,15 and 490, and 'python' in document 1 only. Let's say that the corresponding positional index is like {'coding': [2, {1:[10, 28], 36:[67, 93]}], 'in': [3, {1:[11,46], 15:[5,31,98], 490:[197, 223]}], 'python': [1, {1:[12, 19, 127]}] ...}, it means that the term 'coding' appears in 2 documents (given by the first element of the list): in document 1 it appears in positions 10 and 28, and in document 36 in positions 67 and 93 ; the  term 'python' appears in the first document in positions 12,19 and 127, the same interpretation goes for the term 'in'. In this example of positional index, only the document frequencies are given (2 for the term 'coding' in the precious example) but the raw frequencies are not given (for the term 'coding' it would have been 2 for both documents 1 and 36).

- posting list or posting: is a (generally sorted) list of documents id, i.e a value of an inverted index key. In the example of inverted index given above, [1,36] is the posting list for 'coding'.

# Searching

The pipeline of the search engine goes as follows (for now):

- Preparing the search engine:

    - the collection of documents is loaded via the script loadcollection.py (e.g. load_collection_cran() function for cran dataset which is first stored on disk inside the folder named data, a direct load from a repository could be considered)
    - the documents are iteratively preprocessed using preprocess.py file, indexed with the index_collection() function from collection.py file and the index_document() method associated to the type of index (inverted index or positional index for example in the indices.py file), IDs are at the same time given to the documents.
    - the index along with other data (n-grams of token, ...) are saved on disk (saveload.py file)

- Running the search engine (look at the basic_search_engine.ipynb file):

    - once the user issues a query, a given model takes it as input
    - the query is preprocessed depending on the index used by the model (preprocess.py file)
    - some computations are done (depending on the model) in order to retrieve relevant documents (search.py file)
