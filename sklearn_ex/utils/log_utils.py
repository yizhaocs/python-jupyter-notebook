import os
import logging
from logging.handlers import SysLogHandler

debug = eval(os.environ.get("PHANOMALY_DEBUG", "False"))
log_level = logging.DEBUG if debug else logging.INFO


class TaskLogFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from celery._state import get_current_task
            self.get_current_task = get_current_task
        except ImportError:
            self.get_current_task = lambda: None

    def format(self, record):
        task = self.get_current_task()
        if task and task.request:
            record.__dict__.update(task_id=task.request.id,
                                   task_name=task.name)
        else:
            record.__dict__.setdefault('task_name', 'main')
            record.__dict__.setdefault('task_id', 'main')
        return super().format(record)


syslog_handler = SysLogHandler(address='/dev/log')
syslog_handler.setLevel(log_level)
fmt = '%(name)s[%(lineno)d]: [PH_ML_APP]:[eventSeverity]=%(levelname)s,[task_id]=%(task_id)s,[task_name]=%(task_name)s,[procDetails]={%(message)s}'
syslog_handler.setFormatter(TaskLogFormatter(fmt=fmt))


def get_logger(file):
    name = os.path.basename(file).split('.')[0]
    ph_logger = logging.getLogger(name)
    ph_logger.setLevel(log_level)
    ph_logger.addHandler(syslog_handler)

    return ph_logger
