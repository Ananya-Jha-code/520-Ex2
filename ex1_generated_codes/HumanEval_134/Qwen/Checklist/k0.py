def check_if_last_char_is_a_letter(txt):
    if not txt:
        return False
    
    last_char = txt[-1]
    
    # Check if the last character is a letter and not part of a word
    if last_char.isalpha() and (len(txt) == 1 or txt[-2].isspace()):
        return True
    else:
        return False