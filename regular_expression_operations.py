import re


def remove_anything_that_is_not_alphanumeric_or_underscore():
    s = "FSM-GFU-Window2012R2-WIN2012R2-172-30-56-123"
    '''
        \w will match alphanumeric characters and underscores
        [^\w] will match anything that's not alphanumeric or underscore
    '''
    s = re.sub(r'[^\w]', '', s)
    return s  # FSMGFUWindow2012R2WIN2012R21723056123


if __name__ == '__main__':
    print(remove_anything_that_is_not_alphanumeric_or_underscore())  # FSMGFUWindow2012R2WIN2012R21723056123
