import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from pythonjsonlogger import jsonlogger
import contextvars

from backend.app.config import settings

# Context variables for injecting request scope into logs
request_id_var = contextvars.ContextVar("request_id", default="-")
session_id_var = contextvars.ContextVar("session_id", default="-")

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["app"] = settings.APP_NAME
        log_record["env"] = getattr(settings, "APP_ENV", "development")
        log_record["request_id"] = request_id_var.get()
        log_record["session_id"] = session_id_var.get()

def setup_logger():
    logger = logging.getLogger(settings.APP_NAME)
    
    # Don't propagate or duplicate handlers if already set
    if logger.handlers:
        return logger

    # In production use INFO, otherwise DEBUG
    log_level = logging.INFO
    if getattr(settings, "APP_ENV", "development") == "development" and settings.DEBUG:
        log_level = logging.DEBUG
        
    logger.setLevel(log_level)
    logger.propagate = False

    formatter = CustomJsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(filename)s %(lineno)d %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler (Daily rotation, max size handling isn't built into TimedRotating naturally but we can use backupCount)
    log_dir = settings.BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Rotate at midnight, keep 30 days of logs
    file_handler = TimedRotatingFileHandler(
        filename=log_dir / "app.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()
