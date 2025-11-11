def get_sum(num):   # helper function to calculate sum of digits   
     return sum([int(digit) for digit in str(abs(num))])     
         
        
 def order_by_points(lst):       
       if len(lst)!=0: 
           sortedList =sorted ( lst, key =  get_sum )   # sorting list based on sum of digits.   
            for i in range((len(get_sum)),1,-1) :     # rearranging the elements if required by question statement     
                index=lst.index(sortedList[i])       # getting original indices  and storing them  
                 lst[i], sortedList[-i-1] = ( srotedList[-i -1 ], lst [ i ] )    # swapping elements at their positions in list    
           return sortedList     
        else:         
            return []