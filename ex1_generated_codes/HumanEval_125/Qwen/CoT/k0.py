def split_words(txt):
    if ' ' in txt:
        return txt.split()
    elif ',' in txt:
        return txt.split(',')
    else:
        # Count lowercase letters with odd order in the alphabet
        count = 0
        for char in txt:
            if char.islower():
                # Calculate the order of the character (a=0, b=1, ..., z=25)
                order = ord(char) - ord('a')
                if order % 2 == 1:
                    count += 1
        return count

# Examples
print(split_words("Hello world!"))  # ➞ ["Hello", "world!"]
print(split_words("Hello,world!"))  # ➞ ["Hello", "world!"]
print(split_words("abcdef"))       # ➞ 3