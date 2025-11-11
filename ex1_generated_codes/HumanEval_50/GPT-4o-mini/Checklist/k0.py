def shift(s):   # Function to encode/shift characters in a character string by shifting them upwards.
    return "".join([chr((ord(c) - ord('a') + 5) % 26 + ord('a')) for c in s])    
        
            
def shift_backward (s): # Function to decode/shift characters back by shifting them downwards.  
    return "".join([chr((ord(c) - ord('a') -5 ) % 26 + ord('a')) for c in s])