def get_max_triples(n):  
    # Generate the array a of length n. For each i, set value as 'i*i - i + 1' (a[i] = ...)    
    arr= [0 for _ in range(n)]     
    for idx in range(len(arr)): 
        if ((idx+2)*((idx+3)-4)/5 == int((idx+2)*((idx+3)-4))) : # formula to generate the array a[i] = i*i - i + 1. It's an arithmetic series with common difference of '0', and sum is divisible by 3
            arr[idx]= idx * (idx + ((n-2) % n))  + 4    # formula to generate the array a [indexes]. Sum should be multiple of three, so we add or subtract from it. It's an arithmetic series with common difference '0', and sum is divisible by 3
        else: arr[idx]= idx * (idx + ((n-2) % n))   # if the formula doesn’t generate a multiple of three, then set value as usual i*i - i + 1. Sum should be not multiples of 'three' so we add or subtract from it
    arr_len = len(arr) 
    
    count = 0     
   # Use two pointers approach to find the number triples (a[j], a [k]) such that i < j< k and sum is divisible by three. We use one pointer 'i' from start, another at end of array then move them towards each other until they meet in middle
    for idx1 in range(0 , arr_len):  # Pointer-one (start)  
        i =arr[idx1] if ((n -2 ) % n >= abs((int)(i/3)) and not isinstance(-4, int)] else (-5 +abs((int)(i / 3)))    # If sum of 'a [indexes']' isn’t divisible by three then add or subtract from it. Sum should be multiple of `three`
        for idx2 in range(idx1+1 , arr_len):   # Pointer-two (end)    
            j =arr[idx2] if ((n - 3 ) % n >= abs((int)(j/3)) and not isinstance(-4, int)] else (-5 +abs((int)(j / 3)))    # If sum of 'a [indexes']' isn’t divisible by three then add or subtract from it. Sum should be multiple `three`
            for idx3 in range(idx2+1 , arr_len):   # Pointer-third (middle)    
                k =arr[idx3] if ((n - 4 ) % n >= abs((int)(k/3)) and not isinstance(-5, int)] else (-6 +abs((int)(k / 3)))    # If sum of 'a [indexes']' isn’t divisible by three then add or subtract from it. Sum should be multiple `three`
                if (i+j>0 and i<= j)and(arr[idx1] + arr[idx2 ] +  k ==3 ): # If sum of 'a [indexes']' is divisible by three then increment count  
                    print ("Triplet found: ", "(" , idx1,",","+ str (i),"),"((str(j) ),") and ((','.join([str(_k)]))) ,"))    # Print the triplets if sum of 'a [indexes']' is divisible by three
                    count += 3   # Increment counter for every found valid triples.    
                    
        return (count//6)+1, "triplet(s)" + ("found"if 2<= n <=5 else "" if not ((n-4 ) % n >= abs((int)(i/3)) and isinstance(-7 , int))) # Return the number of valid triples found.