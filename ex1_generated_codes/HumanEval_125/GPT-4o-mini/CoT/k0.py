import re
def split_words(txt):  
    if not any('.' in s for s in txt.split()): # no whitespace found, so check commas 
        return sum([1 for c in list(filter(str.islower, filter(str.isalpha, [s[0]+x for x in range(len(txt)) if re.match('^' + s[:-2], txt)]])) # count lowercase letters with odd order starting from 'a', till 'z'.
    else: 
        return [[i , j ]for i,j in enumerate([s[0]+x for x in range(len(txt)) if re.match('^' + s[:-2], txt)]) # split on whitespace or comma found