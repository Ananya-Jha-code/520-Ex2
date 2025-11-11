def sumOfDigits(n):   # helper function to calculate the digit's sum.
    s = 0             # initialize a variable 'sum'.
    while n > 0 :      # loop until there are no more digits left in number ‘num’ .
        rem = n % 10     # get remainder when num is divided by 10 (last digit of the given integer). This will be added to sum.  
                        # and then we remove lastDigit from originalNum, so that it can go on for next iteration until all digits are processed .   
        s += rem         # add current remainder into 'sum' variable ‘s’ , which is initially zero (i.e., the total sum of digit in number).  
                        # and then we move to process remaining part, so that it can go on for next iteration until all digits are processed .   
        n = int(n /10)     # remove lastDigit from originalNum , which is done by dividing num (i.e., the given integer )by 10 and then convert result to an Integer type, so that it can go on for next iteration until all digits are processed .
    return s            # finally we will get sum of digit in number as a final output from this function which is initially zero , i.e., total sum of the given integer's lastDigit (i.e., each time when loop runs, it adds one more to 'sum').  So at end our result would be all digits added up and then sorted based on their index in original list .
    
def order_by_points(numbers):   # main function which will sort the given number according sum of its digit.   
        return [v[0] for v in sorted([(sumOfDigits(abs(x)), i, x) for (i, x) in enumerate(numbers)] ]  # using list comprehension and lambda to get each item from the resultant tuple which is a combination of sum's digit count , index & original number.
        sorted() function will sort this according first element then second if they are same else third (index). So finally we can return only indexes as output for given problem statement .  It means that it would be in ascending order based on the points mentioned above and also maintain their relative positions within list when sum of digits is equal.