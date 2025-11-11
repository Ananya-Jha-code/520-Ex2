import re

def split_words(txt):  
    # Check for white spaces or comma's and return list of words if found, else the number of upper case characters starting from 'A'->'Z'.    
    txt.index(' ')         
    splitted = [word for word in re.split(r'\W+',txt)]   # Split on whitespace and commas using regex  \w matches any alphanumeric character or underscore, + means one or more of the preceding element   
    
    txt.index(',')         
    splitted = [word for word in re.split('[^A-Za-z0-9]+',txt)]  # Split on commas using regex, ' + means one or more of the preceding element  
    return sum(1 for c in txt if c.islower() and (ord(c) - ord('A')) % 2 != 0 )    # Count upper case characters starting from A->Z, check order is odd or not using modulo operation  
