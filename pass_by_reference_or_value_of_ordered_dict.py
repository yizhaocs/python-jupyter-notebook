# Python code to demonstrate
# call by reference
# OrderedDict是mutable type argument, 所以是pass by reference, 能被callee更改value
from collections import OrderedDict


def add_more(dict):
    dict.update({'new': 9999})
    print("Inside Function", dict)


def move_to_end(dict, key):
    dict.move_to_end(key)


if __name__ == '__main__':
    dict = OrderedDict()
    dict.update({'first': 1, 'second': 2})

    add_more(dict)
    print("Outside Function:", dict)

    move_to_end(dict, 'first')
    print("Outside Function:", dict)
