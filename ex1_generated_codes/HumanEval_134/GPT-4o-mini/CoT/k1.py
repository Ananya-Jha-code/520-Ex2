def check_if_last_char_is_a_letter(txt):  
    if len(txt) == 0: # If string txt length = 0 then return False as there are no characters to process.
        return False    
        
    word = ''.join([i for i in list(filter(str.isalnum, txt))])  # filter out non-alphabetical character and join them into a single sentence (word)  
      
    last_char=''
      if len(txt)!= 0:    
        last_char = word[-1]        
          
          return ((last_char.isalpha())  # check whether the final char is an alphabetical character or not, and also it's part of another letter in case when its a vowel/consonant (i e., isn’t last).  
      else:    
        print("The string txt has no characters to process.")   
          return False  # if there are still non-alphabetical character then returns false.