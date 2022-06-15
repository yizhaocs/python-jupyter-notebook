import concurrent
import time
from concurrent.futures import ThreadPoolExecutor


def try_my_operation(second):
    try:
        time.sleep(second)
        print(f"slept for {second}")
    except:
        print('error')


if __name__ == '__main__':
    seconds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    executor = concurrent.futures.ProcessPoolExecutor(len(seconds))
    futures = [executor.submit(try_my_operation, s) for s in seconds]
    concurrent.futures.wait(futures)
