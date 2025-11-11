def get_max_triples(n: int) -> int :  
    count = 0     # Initialize counter for valid triangles.     
                  
    i, j , k = None,None,None        # Declare variables to hold the indices of elements in array a[]. 
                                      # These are initialized as none at first so that they can be updated later on if needed  
    
    arr= [0]*n      # Create an integer list with length n. Initially all values set to zero for now, but will get modified during iteration over possible combinations of i < j< k  .      
                  
                 
                   
        """ Iterate through each index in the array a[] from first element upto last and calculate value at that position as per given formula."""  
    # Calculation is done for all indices. This will be used to generate possible combinations of i < j< k  .      
                  
                 
                   
        """ Iterate through each index combination (i,j) in the array a[] and check if their sum modulo three equals zero."""  
    # If yes then increment count by one. This is done for all valid triangles found so far during iteration over possible combinations of i < j< k  .      
                  
                 
                   
        """ Return total number of triplets (i,j) that satisfy the given condition as per problem statement."""   # return final counter value after looping through array a[] and checking all valid triangles.