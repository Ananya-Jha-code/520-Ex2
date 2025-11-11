def largest_divisor(n: int) -> str : #changed the function's type hint from 'int'->str, as we want to output a string not an integer in our case 
    """ For given input n find all factors of that number and return maximum factor. This is tricky part will be ignored for final implementation but here it serves purpose."""  
    
    # Initialize the list with first divisor which would always equal '1' as per definition, no need to check further 
    largest_divisors = [i for i in range(2 , int(n**0.5)) if n % i == 0 ]  
    
    # Append the larger factors after finding smaller ones because a number can have more than one divisor, so we need to find all of them 
    largest_divisors += [i for i in range(largest_divisors[-1] , int(n/2) + 1 , -1 ) if n % i == 0 ]  
    
    # Return the maximum divisor found. If there is no such number, return 'None' 
    largest = max(largest_divisors or [0])     
        
    print("Largest Divisible Number: ", int(n) if n % (1 + largest ) == 0 else "No Larger divisor found") #prints the number only when it's a valid result, otherwise prints 'None'.  This is to make sure we are not returning an empty string or None.
    return str(largest_divisors[-1]) if n % (1 + largest ) == 0 else "No Larger divisor found" #return the maximum number only when it's a valid result, otherwise returns 'None'.  This is to make sure we are not returning an empty string or None.