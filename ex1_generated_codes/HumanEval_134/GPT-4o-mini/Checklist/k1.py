def check_if_last_char_is_a_letter(txt):  
    if len(txt) == 0 : # Edge case for empty string, as per requirements.
        return False    
        
    lastChar = txt[-1]     
      
    isLetter=False 
          
    alphabetList =  ['A', 'B','C' , ...,'Z']  
            
    if (lastChar in alphabetList): # Checking for letter character.    
        return True             
        
    else:                          # If last char is not a Letter, it must be part of word so returning False  .     
       return False