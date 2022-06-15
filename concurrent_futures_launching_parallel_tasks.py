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
        seconds = [1, 2, 3]
        executor = futures.ProcessPoolExecutor(len(seconds))
        task_set = set(executor.submit(try_my_operation, s) for s in seconds)
        futures.wait(task_set, timeout=None, return_when=futures.ALL_COMPLETED)
        results = list({f.result()["second"]: f.result()["result"]} for f in task_set)
        print(results)
    except:
        print('error')
    finally:
        executor.shutdown()
