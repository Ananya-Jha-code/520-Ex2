def largest_divisor(n: int) -> int:
    """ For a given number n, find the largest number that divides n evenly, smaller than n
    >>> largest_divisor(15)
    5
    """
    for d in range(n-1, 0, -1):
        if n % d == 0:
            return d

# Example usage:
print(largest_divisor(15))  # Output: 5