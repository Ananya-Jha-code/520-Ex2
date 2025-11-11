def check_if_last_char_is_a_letter(txt):
    if not txt or not txt.strip():
        return False
    
    words = txt.split()
    last_word = words[-1]
    
    if len(last_word) > 0 and last_word[-1].isalpha():
        return True
    else:
        return False