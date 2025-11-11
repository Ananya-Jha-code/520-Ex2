def largest_divisor(n: int) -> int:
    """ For a given number n, find the largest number that divides n evenly, smaller than n """
    for i in range(n - 1, 0, -1):
        if n % i == 0:
            return i

# Test cases
assert largest_divisor(15) == 5
assert largest_divisor(28) == 14
assert largest_divisor(37) == 1  # 37 is a prime number