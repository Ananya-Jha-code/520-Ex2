def largest_divisor(n: int) -> int:
    """ For a given number n, find the largest number that divides n evenly, smaller than n
    >>> largest_divisor(15)
    5
    """
    # Initialize the largest divisor as 1
    largest = 1
    
    # Iterate from n-1 down to 1
    for i in range(n - 1, 0, -1):
        # Check if i divides n evenly
        if n % i == 0:
            # Update the largest divisor
            largest = i
            break  # Exit the loop as we found the largest divisor
    
    return largest

# Test cases
print(largest_divisor(15))  # Output: 5
print(largest_divisor(28))  # Output: 14
print(largest_divisor(17))  # Output: 1 (since 1 is the only divisor smaller than 17)