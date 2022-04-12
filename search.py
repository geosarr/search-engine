from IPython.display import clear_output
from search_models import Boolean, Vsm, Bim, BimExt, QueryLklhd, W2Vsm, Lsi




def searching(modl):
    # models=[Boolean, Vsm, Bim, BimExt, QueryLklhd, W2Vsm, Lsi]

    # if type(modl) not in models:
    #     raise ValueError ("The only supported values for the argument model are of type {}".format(list(models)))

    # Thanks to Gael Guibon (https://gitlab.com/gguibon) for this tip :)
    query = input('Welcome to the best search engine ever :)\nEnter some keywords or type "exit" to stop the engine: ')
    while query != "exit":
        results=modl.retrieval(query=query)
        print("Your query: ", query)
        print("\nResult(s): ")
        if results==None: 
            print('Quitting the search engine')
            break

        elif len(results[:modl.top])>0:
            for ID,_ in results[:modl.top]:
                print(modl.index.documents[ID].content,'\n')
        else:
            print('No result found. Try another query :)')
        query = input('Enter some keywords, type "exit" to quit: ')
        clear_output(wait=True)





    
    
    


    
    
    

