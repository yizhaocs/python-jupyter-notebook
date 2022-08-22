# Python code to demonstrate
# call by reference
# OrderedDict是mutable type argument, 所以是pass by reference, 能被callee更改value
from collections import OrderedDict


def add_more(dict):
    dict.update({'new': 9999})
    print("Inside Function", dict)


def remove(dict, key):
    dict.pop(key)


def move_to_end(dict, key):
    dict.move_to_end(key)


def move_to_front(dict, key):
    '''
        For Python 3.2 and later, you should use the move_to_end method.
        The method accepts a last argument which indicates whether the element will be moved to the bottom (last=True)
        or the top (last=False) of the OrderedDict.
    '''
    dict.move_to_end(key, last=False)


if __name__ == '__main__':
    dict = OrderedDict()
    dict.update({'first': 1, 'second': 2})

    add_more(dict)
    print("Outside Function:", dict)

    move_to_end(dict, 'first')
    print("Outside Function:", dict)

    move_to_front(dict, 'first')
    print("Outside Function:", dict)
