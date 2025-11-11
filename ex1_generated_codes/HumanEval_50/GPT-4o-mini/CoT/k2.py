def encode_shift(s):
    """returns encoded string with shift"""  
     return "".join([chr((ord(c) - 97 + 5) % 26 + 97) if 'a' <= c <= 'z' else chr((ord(c) - 391 )% 40 + ord('A')) for c in s])
    
def decode_shift(s):   # Decoding function, shifts back by the same amount to get original string.   
      return "".join([chr((ord(x)-5) %26 + 97 ) if 'a' <= x<= 'z' else chr (( ord (x)) - 34 +  108)] for x in s])     # ASCII values are adjusted as per requirement.