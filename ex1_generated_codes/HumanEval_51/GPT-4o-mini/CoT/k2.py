def remove_vowels(text):    
    vowel = ['a', 'e', 'i', 'o','u']  # list containing all lowercase and upper case English vowels.  
      
    text=' '.join([char for char in text if not (char in vowel)])#list comprehension to filter out the non-vowal characters from input string, then join them back into a single sentence with spaces between each character  # list of all lowercase and upper case English letters.
    
    return(text[:len(text)-1] if text[-1].isalpha() else text)#returning the final result: removing last newline (\n), or returning input string in its original form, as it is not a sentence ending with vowel (i.e., non-alphabetic character).