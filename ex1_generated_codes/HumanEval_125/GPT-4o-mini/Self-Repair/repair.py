def split_words(txt):  
    if "," not in txt and " " not in txt: # no commas or whitespace found, so calculate the number of upper case alphabets with odd order.
        return sum([1 for chr in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if (ord(chr) - ord('A')) % 2 == 0]) # subtract ASCII value from A to get the position of upper case alphabets with odd order.
    elif "," in txt:   # split on commas, return list after splitting by comma first and then join them back together using space as separator if there are multiple words separated by a comma 
        splitted = [word for word in re.split(', ',txt)]    
        result=[]   
        while '' in splitted:   # remove empty strings from list, this will happen when the string is split on commas and has extra spaces or no space between words separated by comma 
            if "" not in splitted : break     
             else:splitted.remove("")    
                
        return result + [word for word in re.split(' ',txt)] # join remaining strings with a whitespace as separator and add them to the list  
    elif " " in txt:# split on spaces, same process above 
         splitted =[word for word in re.split(" ",txt) ]    
        result=[]     
          while '' in splitted: # remove empty strings from a given text string if there are multiple words separated by space or no whitespace between them  
            if "" not in splitted : break      
             else:splitted.remove('')   
                 return  [word for word in re.split(', ',txt)] + result # join remaining strings with a comma as separator and add to the list