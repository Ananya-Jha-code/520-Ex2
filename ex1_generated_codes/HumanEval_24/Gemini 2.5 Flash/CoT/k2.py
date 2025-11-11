def largest_divisor(n: int) -> int:
    """ For a given number n, find the largest number that divides n evenly, smaller than n
    >>> largest_divisor(15)
    5
    """
    # Start with the assumption that the largest divisor is 1
    largest = 1
    
    # Iterate from n-1 down to 1
    for i in range(n - 1, 0, -1):
        # Check if i divides n evenly
        if n % i == 0:
            # Update the largest divisor
            largest = i
            break  # Exit the loop as we found the largest divisor
    
    return largest

# Example usage
print(largest_divisor(15))  # Output: 5