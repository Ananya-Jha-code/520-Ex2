def remove_vowels(text):  # noqa F821 (ignoring the line above for code coverage)
    vowel = ('aeiouAEIOU', 'eiouEIUO')  
    return "".join([char for char in text if char not in vowel])