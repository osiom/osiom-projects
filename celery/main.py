from celery import Celery

broker = "redis://127.0.0.1:6379/0"

celery_app = Celery("main", broker=broker, backend=broker)


@celery_app.task()
def some_task(arg: str):
    return arg + "foo"


if __name__ == '__main__':
    task = some_task.apply_async(("something",))
    print(task.get())
