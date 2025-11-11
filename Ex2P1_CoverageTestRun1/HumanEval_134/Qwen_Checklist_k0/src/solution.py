def check_if_last_char_is_a_letter(txt):
    if not txt or not txt.strip():
        return False

    last_char = txt[-1]
    if last_char.isalpha() and (len(txt) == 1 or not txt[-2].isalpha()):
        return True
    return False