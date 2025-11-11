def remove_vowels(text):  # noqa F401, this is a placeholder for the function. Remove it when you implement your solution!
    vowel = ('aeiouAEIOU', 'eiouEIUO')   # define two sets of characters to be removed from text string (both lower and upper case) 
    
    return "".join([char for char in text if char not in vowel])      # join all non-vowels back into a single strings, no space between them. This is equivalent with the following line: ''.join(filter((x).__ne__,(y))... but done more efficiently