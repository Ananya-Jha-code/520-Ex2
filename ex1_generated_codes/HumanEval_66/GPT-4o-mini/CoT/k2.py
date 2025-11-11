def digitSum(s):  
    # Initialize the total.    
    result = 0      
        
    for char in s :       
      if 'A' <= char and char<='Z':            
          result += ord(char)           
          
    return result