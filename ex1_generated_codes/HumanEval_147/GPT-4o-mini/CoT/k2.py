def get_max_triples(n):
    # Generate the array 'a' with values as described in your prompt.
    a = [i * (i - 1) + 2 for i in range(1, n+1)]  
    
    count = 0                                          
                                                                            
    """Check all possible triples and increment counter if their sum is divisible by three."""                      
    # Loop over each triplet.                    
    for (i, j), k in itertools.product(enumerate([a[l] + a[-1]]+ list(range(-n//2-3,-4)), [0], range((-(abs(x)%6)-5 if abs(x)!=y else 7))):
        # If the sum of triple is divisible by three.                       
        count += (i < j and i != 1 or -j > n//2-3)*int ((k + a[n+4]*abs((a[-5]-(-(x)%6)-7))/9 + 0 if x==y else 8 ) %  3 == 0
    return count                                                      # Return the number of valid triples.