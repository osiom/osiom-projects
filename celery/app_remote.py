# This app won't print anything but just send tasks
from celery import Celery

broker = "redis://127.0.0.1:6379/0"
celery_app = Celery("main", broker=broker, backend=broker, include=["tasks"])

def calculate_sum_app():
    """ Adds task to the queue with apply_async method.
    the method doesn't wait the task execution be finished.
    """
    task_calculate_sum = celery_app.send_task("main.calculate_sum", args=([25,25]))
    print(f"task is running with the id: {task_calculate_sum.task_id}")


def calculate_string_app(text: str):
    """ Adds task to the queue with apply_async method.
    the method doesn't wait the task execution be finished.
    """
    task_calculate_string = celery_app.send_task("main.calculate_string", args=(text, ))
    # get information about the task created
    print(f"task is running with the id: {task_calculate_string.task_id}")
    print(f"task status is: {task_calculate_string.status}")
    print(f"status result of the task is: {task_calculate_string.result}")
    print("--We are now gonna wait for the result to succeed, not async task--")
    result = task_calculate_string.get() # get the result of the task
    print(f"the result of the task itself is: {result}")
    print(f"status changed now is: {task_calculate_string.status}")

print("*** RUNNING CALCULATE SUM TASK ***")
calculate_sum_app()
print("*** RUNNING CALCULATE STRING TASK ***")
calculate_string_app(text="Hello")
print("*** RUNNING TEST ON EXCEPTION ***")
exception_test.send_task("main.something", args=())
