def check_if_last_char_is_a_letter(txt):  
    # Remove any non-alphabetical characters from txt.
    cleaned = ''.join([c for c in txt if c.isalpha()])    
        
    return len(cleaned) > 0 and (not bool(re.search('[A-Za-z]', cleaned)))  # Return True only when there is at least one non alphabetical character, or the last char was not an uppercase letter/digit from A to Z in either case