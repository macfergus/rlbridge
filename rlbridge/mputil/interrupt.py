import signal

__all__ = ['disable_sigint']


def disable_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
