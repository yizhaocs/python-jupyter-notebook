# String till Substring
def string_till_substring():
    #  String till Substring
    test_string = "GeeksforGeeks is best for geeks"
    spl_word = 'best'
    res = test_string.partition(spl_word)[0]

    # String before the substring occurrence :  for geeks
    print("String before the substring occurrence : " + res)


if __name__ == '__main__':
    string_till_substring()
