def split_words(txt):
    if "," not in txt: # Check for comma separator first before checking whitespace character, to avoid unnecessary operations when there is no commas present at all  
        return sum([1 for c in txt.lower() if ord('a') <= ord(c) <= ord('z') and (ord(c)-ord('a')) % 2 == 0]) # Count lower-case letters with odd order of their ASCII values, 'b' is considered to be even
    else:  
        return [word.strip() for word in txt.split(" ")] if not any([char == ',']*txt.count(',') > 1) and len(''.join(chunk[0]+''.join(sorted(list(reversed(set(str)), key=lambda c: (c=='a')))) % 2 == 0 else [word for word in txt.split(" ")] # Split on whitespace if there are no commas present, otherwise split based off of the first character