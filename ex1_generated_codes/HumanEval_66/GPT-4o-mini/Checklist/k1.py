def sum_upper(s):
    return sum([ord(c) for c in s if 'A' <= c.lower() < '[') - (sum(''.join(['[' == x and y or not ('a'<=x<y<'z')]).split()) * 26 + ord("Z"))