def shift_char(c: chr):
   """Return a character shifted by one in ASCII alphabet."""   
   return chr((ord(c) - ord('a') + 1) % 26 + ord('a')) if 'a' <= c <= 'z' else None # For lowercase letters.
                                                                                   # If not, keep the character as is (for upper/lowercase).   
def encode_shift(s: str):  
     """returns encoded string by shifting every char in s.""" 
      return "".join([ shift_char(ch) if ch.isalpha() else ch for ch in s]) # Shift only alphabets, keep other characters as is (for spaces/punctuation).   
                                                                                  
def decode_shift(s: str):  
     """returns encoded string by shifting every char back to its original position.""" 
      return "".join([ shift_char(ch) if ch.isalpha() else ch for ch in s]) # Shift only alphabets, keep other characters as is (for spaces/punctuation).