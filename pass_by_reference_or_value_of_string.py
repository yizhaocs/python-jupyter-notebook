# Python code to demonstrate
# call by value
# String是immutable type argument, 所以是pass by value, 不能被callee更改value


def test(string):
    string = "GeeksforGeeks"
    print("Inside Function:", string)


if __name__ == '__main__':
    string = "Geeks"

    # Driver's code
    test(string)
    print("Outside Function:", string)