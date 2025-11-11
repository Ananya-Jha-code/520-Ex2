def words_string(s):
    return s.replace(' ', '').split(',') if not any([c in ['{', '['] for c in s]) else [word.strip() for word in re.findall("'''([^']+)'''", s)]  # noqa: E501