# Python code to demonstrate
# call by reference
# Dict是mutable type argument, 所以是pass by reference, 能被callee更改value

def add_more(dict):
    dict.update({'new': 9999})
    print("Inside Function", dict)


if __name__ == '__main__':
    dict = {'first': 1, 'second': 2}

    add_more(dict)
    print("Outside Function:", dict)