from time import sleep
from celery import Celery

broker = "redis://127.0.0.1:6379/0"

celery = Celery("main", broker=broker, backend=broker)
print(celery)

class FooException(Exception):
   def __init__(self, msg):
       print(msg)
       print("INTIATED IN WORKER")

@celery.task
def exception_test():
   raise FooException("EXCEPTION RAISED")

@celery.task()
def calculate_string(arg: str):
    sleep(10)
    return arg + " World"

@celery.task()
def calculate_sum(arg_one: int, arg_two: int):
    return arg_one + arg_two
