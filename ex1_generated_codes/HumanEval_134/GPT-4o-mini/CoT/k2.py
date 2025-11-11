def check_if_last_char_is_a_letter(txt):  
    txt = ''.join([i for i in (''.join(filter(str.isalnum, s)) if not s else '' for s in [reversed(txt)]]) # remove non-space characters and reverse string 
    
    return bool((len(txt) > 0 and ord('a') <= ord(txt[0]) <= ord('z')) or (ord('A')<= ord(txt[0]))