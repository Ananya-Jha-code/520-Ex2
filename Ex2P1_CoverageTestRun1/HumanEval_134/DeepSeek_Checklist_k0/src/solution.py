def check_if_last_char_is_a_letter(txt):  
    if not txt or len(txt) == 0 : # edge case when string is empty/null.
        return True    
        
    words = [word for word in ' '.join(''.join([chr, chr])).split()] 
      
    last_char  = list(reversed(list((txt[-1]))))   # get the reversal of string and convert it into a character.    
        
      return all (c.isalpha() for c in words[::-1].extend([last_char]) if len(words) > 0 else [False] )