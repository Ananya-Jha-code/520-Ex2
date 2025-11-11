def split_words(txt):
    # Check if the string contains whitespace
    if ' ' in txt:
        return txt.split()
    
    # Check if the string contains commas
    elif ',' in txt:
        return txt.split(',')
    
    else:
        # Calculate the number of lowercase letters with odd order in the alphabet
        count = 0
        for char in txt:
            if char.islower():
                # Get the ASCII value of the character and subtract 97 to get its position in the alphabet
                pos = ord(char) - 97
                # Check if the position is odd
                if pos % 2 == 1:
                    count += 1
        return count

# Example usage:
print(split_words("Hello world!"))  # ➞ ["Hello", "world!"]
print(split_words("Hello,world!"))  # ➞ ["Hello", "world!"]
print(split_words("abcdef"))      # ➞ 3