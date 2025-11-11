def encode_shift(s: str):
    """returns encoded string by shifting every character by 5 positions right"""  
     return "".join([chr((ord(ch) - ord("a") + 5) % 26 + ord("a")) for ch in s]) # Shift to the left, so we subtract from 'z' (ASCII value of z). If you want shift right then add.
     
def decode_shift(s: str):  
    """takes as input string encoded with encode function and returns original/decoded""" 
     return "".join([chr((ord(ch) - ord("a") -5 ) %26 + ord('a')) for ch in s]) # Shift to the left, so we subtract from 'z' (ASCII value of z). If you want shift right then add.