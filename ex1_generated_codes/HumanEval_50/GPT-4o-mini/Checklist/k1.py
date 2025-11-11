def shift(s):   # Function for encoding and then shifting characters by '5' places.
    return "".join([chr((ord(c) - ord('a') + 5) % 26 + ord('a')) if ('a' <= c <= 'z') else chr((ord(c) - ord('A') + 5) % 26 + ord('A')) for c in s])
    
def decode_shift(s):   # Function to shift characters back by '-5'. Useful when you want the original text.
    return "".join([chr((ord(c) - ord('a') - 5) % 26 + ord('a')) if ('a' <= c <= 'z') else chr((ord(c) - ord('A') - 5) % 26 + ord('A')) for c in s])