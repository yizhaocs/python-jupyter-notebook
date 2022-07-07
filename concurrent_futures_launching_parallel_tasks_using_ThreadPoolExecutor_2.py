import time

from concurrent import futures


def try_my_operation(second):
    try:
        print(f"going to sleep for {second}")
        time.sleep(second)
    except:
        print('error')
    return {"second": second, 'result': f"slept for {second}"}


if __name__ == '__main__':
    try:
        executor = futures.ThreadPoolExecutor(3)
        future1 = executor.submit(try_my_operation, 1)
        future2 = executor.submit(try_my_operation, 2)
        future3 = executor.submit(try_my_operation, 3)

        futures.wait([
            future1,
            future2,
            future3], timeout=None, return_when=futures.ALL_COMPLETED)
        print(future1.result())
        print(future2.result())
        print(future3.result())
    except:
        print('error')
    finally:
        executor.shutdown()
