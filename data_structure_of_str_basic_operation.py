# String till Substring
def string_till_substring():
    #  String till Substring
    test_string = "GeeksforGeeks is best for geeks"
    spl_word = 'best'
    before_spl_word = test_string.partition(spl_word)[0]
    after_spl_word = test_string.partition(spl_word)[2]

    # String before the substring occurrence : GeeksforGeeks is
    print("String before the substring occurrence : " + before_spl_word)
    # String after the substring occurrence :  for geeks
    print("String after the substring occurrence : " + after_spl_word)


if __name__ == '__main__':
    string_till_substring()
