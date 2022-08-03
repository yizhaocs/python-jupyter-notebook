# Python code to demonstrate
# call by reference
# List是mutable type argument, 所以是pass by reference, 能被callee更改value


def add_more(list):
    list.append(50)
    print("Inside Function", list)


if __name__ == '__main__':
    mylist = [10, 20, 30, 40]

    add_more(mylist)
    print("Outside Function:", mylist)