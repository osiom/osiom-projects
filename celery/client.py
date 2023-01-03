from celery import Celery

broker = "redis://127.0.0.1:6379/0"

celery_app = Celery("splitting_tool", broker=broker, backend=broker)

task = celery_app.send_task("main.some_task", args=("something",))

print(task.task_id)
print(task.status)
print(task.result)

result = task.get()

print(task.task_id)
print(task.status)
print(result)
print(task.result)
