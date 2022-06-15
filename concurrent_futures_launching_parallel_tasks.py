import time

from concurrent import futures


def try_my_operation(second):
    try:
        print(f"going to sleep for {second}")
        time.sleep(second)
    except:
        print('error')
    return f"slept for {second}"


if __name__ == '__main__':
    try:
        seconds = [1, 2, 3]
        executor = futures.ProcessPoolExecutor(len(seconds))
        task_set = set(executor.submit(try_my_operation, s) for s in seconds)
        futures.wait(task_set, timeout=None, return_when=futures.ALL_COMPLETED)
        merged_list = [v for f in task_set for v in f.result()]
        print(merged_list)
    except:
        print('error')
    finally:
        executor.shutdown()
