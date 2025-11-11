def words_string(s):
    return s.replace(' ', '').split(',') if ',' in s else s.split()   # If there is a comma, split by commas; otherwise just use spaces to break up the string into individual elements (words). 
    
# Testing our function with some examples:
print(words_string("Hi, my name is John"))    # Should print ['Hi', 'my', 'name', 'is', 'John']  
print(words_string('One, two, three, four'))  # should also output [‘One’，'two,' ,'three','four'], because there are commas in the string.