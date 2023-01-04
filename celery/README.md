# Celery Playground

This project has been created to have a better understanding and experimenting with Celery.

# Create Broker
## spin up Redis container
    docker-compose up -d

# Useful Celery Commands
## spin up workers
    celery -A <project> worker -P solo -E -l INFO
## get status of nodes
    celery -A <project> status
## list active tasks
    celery -A <project> inspect active
## list registered tasks
    celery -A <project> inspect active

# Run Python code
## Run client
    python3 app.py