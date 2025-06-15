import os
import logging
import platform
import warnings

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource, SERVICE_INSTANCE_ID, SERVICE_VERSION, SERVICE_NAMESPACE
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from configuration import Configuration

# Custom filter to exclude trace logs
class ExcludeTraceLogsFilter(logging.Filter):
    def filter(self, record):
        filter_out = 'applicationinsights' not in record.getMessage().lower()
        filter_out = filter_out and 'response status' not in record.getMessage().lower()
        filter_out = filter_out and 'transmission succeeded' not in record.getMessage().lower()
        return filter_out

class Telemetry:
    """
    Manages logging and the recording of application telemetry.
    """

    log_level : int = logging.WARNING
    azure_log_level : int = logging.WARNING
    langchain_log_level : int = logging.NOTSET
    api_name : str = None
    telemetry_connection_string : str = None

    @staticmethod
    def configure_basic(config: Configuration):
        # User Warning configuration
        
        # Available options for USER_WARNING_FILTER:
        #   ignore  - never show the warning
        #   always  - always show the warning
        #   error   - turn the warning into an exception
        #   once    - show the warning only once
        #   module  - show the warning only once per module
        #   default - default Python behavior
        user_warning_filter = config.get_value('USER_WARNING_FILTER', 'ignore').lower()
        warnings.filterwarnings(user_warning_filter, category=UserWarning)

        level=config.get_value('LOGLEVEL', 'DEBUG').upper()

        #convert to logging level
        if level == 'DEBUG':    
            level = logging.DEBUG
        elif level == 'INFO':
            level = logging.INFO
        elif level == 'WARNING':
            level = logging.WARNING
        elif level == 'ERROR':
            level = logging.ERROR
        elif level == 'CRITICAL':
            level = logging.CRITICAL

        logging.basicConfig(level=level, force=True)
        logging.getLogger("azure").setLevel(config.get_value('AZURE_LOGLEVEL', 'WARNING').upper())
        #logging.getLogger("httpx").setLevel(config.get_value('HTTPX_LOGLEVEL', 'ERROR').upper())
        #logging.getLogger("httpcore").setLevel(config.get_value('HTTPCORE_LOGLEVEL', 'ERROR').upper())
        #logging.getLogger("openai._base_client").setLevel(config.get_value('OPENAI_BASE_CLIENT_LOGLEVEL', 'WARNING').upper())
        #logging.getLogger("urllib3").setLevel(config.get_value('URLLIB3_LOGLEVEL', 'WARNING').upper())
        #logging.getLogger("urllib3.connectionpool").setLevel(config.get_value('URLLIB3_CONNECTIONPOOL_LOGLEVEL', 'WARNING').upper())
        #logging.getLogger("openai").setLevel(config.get_value('OPENAI_LOGLEVEL', 'WARNING').upper())
        #logging.getLogger("autogen_core").setLevel(config.get_value('AUTOGEN_CORE_LOGLEVEL', 'WARNING').upper())
        #logging.getLogger("autogen_core.events").setLevel(config.get_value('AUTOGEN_EVENTS_LOGLEVEL', 'WARNING').upper())
        #logging.getLogger("uvicorn.error").propagate = True
        #logging.getLogger("uvicorn.access").propagate = True


    @staticmethod
    def configure_monitoring(config: Configuration, telemetry_connection_string: str, api_name : str):

        Telemetry.telemetry_connection_string = config.get_value(telemetry_connection_string)
        Telemetry.api_name = api_name
        resource = Resource.create(
            {
                SERVICE_NAME: f"{Telemetry.api_name}",
                SERVICE_NAMESPACE : f"gpt-rag-agentic",
                SERVICE_VERSION: f"1.0.0",
                SERVICE_INSTANCE_ID: f"{platform.node()}"
            })

        # Configure Azure Monitor defaults
        configure_azure_monitor(
            connection_string=Telemetry.telemetry_connection_string,
            disable_offline_storage=True,
            disable_metrics=True,
            disable_tracing=False,
            disable_logging=False,
            resource=resource
        )

        #Configure telemetry logging
        Telemetry.configure_logging(config)

    @staticmethod
    def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        return logger

    @staticmethod
    def get_tracer(name: str) -> Tracer:
        return trace.get_tracer(name)

    @staticmethod
    def record_exception(span: Span, ex: Exception):
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(ex)

    @staticmethod
    def translate_log_level(log_level: str) -> int:
        if log_level == "Debug":
            return logging.DEBUG
        elif log_level == "Trace":
            return logging.DEBUG
        elif log_level == "Information":
            return logging.INFO
        elif log_level == "Warning":
            return logging.WARNING
        elif log_level == "Error":
            return logging.ERROR
        elif log_level == "Critical":
            return logging.CRITICAL
        else:
            return logging.NOTSET

    @staticmethod
    def configure_logging(config: Configuration):

        Telemetry.log_level = Telemetry.translate_log_level(
            config.get_value("Mcp:Logging:LogLevel:Default", default= "Information"))
        
        Telemetry.azure_log_level = Telemetry.translate_log_level(
            config.get_value("Mcp:Logging:LogLevel:Azure", default= "Information"))

        enable_console_logging = config.get_value("FoundationaLLM:PythonSDK:Logging:EnableConsoleLogging")

        handlers = []

        if Telemetry.log_level == logging.DEBUG:
            handlers.append(logging.StreamHandler())

        #Logging configuration
        LOGGING = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
                },
                'standard': {
                    'format': '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
                },
                'azure': {
                    'format': '%(name)s: %(message)s'
                },
                'error': {
                    'format': '[%(asctime)s] [%(levelname)s] %(name)s %(process)d::%(module)s|%(lineno)s:: %(message)s'
                }
            },
            'handlers': {
                'default': {
                    'level': Telemetry.log_level,
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                    'filters' : ['exclude_trace_logs'],
                    'stream': 'ext://sys.stdout',
                },
                'console': {
                    'level': Telemetry.log_level,
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                    'filters' : ['exclude_trace_logs'],
                    'stream': 'ext://sys.stdout'
                },
                "azure": {
                    'formatter': 'azure',
                    'level': Telemetry.log_level,
                    "class": "opentelemetry.sdk._logs.LoggingHandler",
                    'filters' : ['exclude_trace_logs'],
                }
            },
            'filters': {
                'exclude_trace_logs': {
                    '()': 'foundationallm.telemetry.ExcludeTraceLogsFilter',
                },
            },
            'loggers': {
                'azure': {  # Adjust the logger name accordingly
                    'level': Telemetry.azure_log_level,
                    "class": "opentelemetry.sdk._logs.LoggingHandler",
                    'filters': ['exclude_trace_logs']
                },
                '': {
                    'handlers': ['console'],
                    'level': Telemetry.log_level,
                    'filters': ['exclude_trace_logs'],
                },
            },
            "root": {
                "handlers": ["azure", "console"],
                "level": Telemetry.log_level,
            }
        }

        #remove console if prod env (cut down on duplicate log data)
        if enable_console_logging != 'true':
            LOGGING['root']['handlers'] = ["azure"]

        #set the logging configuration
        logging.config.dictConfig(LOGGING)