import time

from concurrent import futures


def try_my_operation(second):
    try:
        time.sleep(second)
        print(f"slept for {second}")
    except:
        print('error')


if __name__ == '__main__':
    seconds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    executor = futures.ProcessPoolExecutor(len(seconds))
    task_list = list(executor.submit(try_my_operation, s) for s in seconds)
    futures.wait(task_list, timeout=None, return_when=futures.ALL_COMPLETED)
