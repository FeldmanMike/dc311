import logging
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'file_handler': {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'filename': 'logs/dev.log',
            'mode': 'a',
            'level': 'INFO',
        },
        'console_handler': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'DEBUG',
        },
    },
    'loggers': {
        '': {
            'handlers': ['file_handler', 'console_handler'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)

