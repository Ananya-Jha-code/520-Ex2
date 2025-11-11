from src.solution import *
import pytest

def test_large_prime_number():
    # Prime number edge case near 1000
    assert largest_divisor(997) == "No Larger divisor found"

def test_power_of_two():
    # Powers of 2 have multiple divisors, max should be n/2
    assert largest_divisor(64) == "32"

def test_input_is_one():
    # 1 has no divisors other than itself
    assert largest_divisor(1) == "No Larger divisor found"

def test_negative_number():
    # Should ideally return "No Larger divisor found" or raise an error
    assert largest_divisor(-20) == "No Larger divisor found"

def test_input_is_composite_with_close_divisors():
    # 77 = 7 * 11; max divisor is 77 / 7 = 11 or 77 / 11 = 7
    assert largest_divisor(77) == "11"
