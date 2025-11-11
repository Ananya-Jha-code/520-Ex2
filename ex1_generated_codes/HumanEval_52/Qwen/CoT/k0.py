def below_threshold(l: list, t: int) -> bool:
    """Return True if all numbers in the list l are below threshold t.
    >>> below_threshold([1, 2, 4, 10], 100)
    True
    >>> below_threshold([1, 20, 4, 10], 5)
    False
    """
    # Check if the list is empty or has only one element
    if not l:
        return True
    
    # Iterate over each element in the list
    for num in l:
        # If any number is not below the threshold, return False
        if num >= t:
            return False
    
    # If all numbers are below the threshold, return True
    return True