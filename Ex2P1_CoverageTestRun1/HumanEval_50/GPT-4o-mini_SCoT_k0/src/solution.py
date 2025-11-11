def shift(s):   # Function that shifts every character by five positions forward within alphabetical order.
    return "".join([chr((ord(c) - ord("a") + 5) % 26 + ord("a")) for c in s])    
        
            
# Testing the function with some inputs:   # You can add more tests here if needed, or delete this section.   
print('Original Text :', 'hello')       
encoded_text = shift(shift('hello'))      
decodingText='' 
for char in encoded_text:                  
     decodeingChar =  chr((ord(char) - ord("a")-5 ) %26 + ord ("a"))   # Decodes the shifted text.   
                                                          print ('Decoded Text :',decodeingChar, end='')