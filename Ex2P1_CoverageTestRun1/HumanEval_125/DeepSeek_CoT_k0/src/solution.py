def split_words(txt):
    # Check for white space character first to decide whether we should use whitespace as a delimiter. 
    if ' ' in txt : splits = [s.strip() for s in re.split(' ', txt)]  
    
    elif ',' in txt: splits =  [s.strip(',')for s in re.split(',',txt) ] # Checking comma first 
        
    else:# if no spaces or commas are found, we calculate the number of lower-case letters with odd order and return it as an integer value  
        count = sum([1 for c in txt if 'a'<=c.lower() <='z']) # Counts all characters that have a letter from "b" to z  (both inclusive)   
        
        lower_odd_count = 0    
      
        odd_chars  = [char for char, count in collections.Counter(txt).items() if int(ord(char)) %2 != 0 and 'a'<=char <='z'] # get all characters with an order of 1 (odds) from "b" to z  
        
        lower_odd_count += sum([1 for char in odd_chars])   
      
     return splits if len(splits)!= 0 else count,lower_odd_count # Return either list or integer value