def check_if_last_char_is_a_letter(txt):
    if not txt:
        return False
    
    words = txt.split()
    last_word = words[-1]
    
    if len(last_word) > 0 and last_word[-1].isalpha():
        return True
    else:
        return False

# Test cases
print(check_if_last_char_is_a_letter("apple pie")) ➞ False
print(check_if_last_char_is_a_letter("apple pi e")) ➞ True
print(check_if_last_char_is_a_letter("apple pi e ")) ➞ False
print(check_if_last_char_is_a_letter("")) ➞ False