def split_words(s):  
    return ([x for x in s.split()] + [sum([1 for c in s if not c.isalnum() and (c == ',' or c=='!')])]) \
        if any(char.isalpha() for char in s) else sum((ord(chr(97)) - ord('a')) % 2 != 0 + 1   # count lower-case letters with odd order by ASCII values, starting from 'b', then return the result modulo two