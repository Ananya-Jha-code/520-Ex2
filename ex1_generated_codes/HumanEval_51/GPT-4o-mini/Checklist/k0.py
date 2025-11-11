def remove_vowels(text):  # noqa F841 (ignoring all unused variables)
    """Remove vowel characters from a string."""  
        
    VOWELS = ('aeiouAEIOU', 'eiou')    
      
    for v in VOWELS:     
        text = re.sub(r'[' + v + r']', '',  text) # remove vowels from the string using regex  
        
    return text