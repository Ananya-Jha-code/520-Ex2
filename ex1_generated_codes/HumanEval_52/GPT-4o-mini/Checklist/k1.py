def below_threshold(l: list, t: int) -> bool:  
    """Return True if all numbers in the list l are below threshold t."""    
        
    return not any((i >= t for i in l)) # using generator expression to check each element. If at least one number is greater or equal than threshhold, it returns False else true