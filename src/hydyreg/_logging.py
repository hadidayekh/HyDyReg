"""Logging of hydyreg: logger "hydyreg". The library is silent unless logging is configured, e.g. with
`hydyreg.verbose()` (INFO: progress and results; DEBUG: details of every segment and start).

info(), debug() and warning() take the arguments of print (joined by spaces) and only build the message when it
will be emitted.
"""
import logging

logger = logging.getLogger("hydyreg")


def _msg(args):
    return " ".join(str(a) for a in args)


def info(*args):
    if logger.isEnabledFor(logging.INFO):
        logger.info(_msg(args))


def debug(*args):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(_msg(args))


def warning(*args):
    logger.warning(_msg(args))


def enabled(level=logging.INFO):
    return logger.isEnabledFor(level)


def verbose(level="INFO"):
    """Shows hydyreg's messages on the console from this level on ('DEBUG', 'INFO', 'WARNING')."""
    level = getattr(logging, level) if isinstance(level, str) else level
    if not any(getattr(h, "_hydyreg", False) for h in logger.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        h._hydyreg = True
        logger.addHandler(h)
    logger.setLevel(level)
