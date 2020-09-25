import warnings

from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import (BaseCallback,
                                                EvalCallback)


class LoggerCallback(BaseCallback):

    def __init__(self, _format, log_on_start=None, suffix=""):
        super().__init__()
        self._format = _format
        self.suffix = suffix
        if log_on_start is not None and not isinstance(log_on_start, (list, tuple)):
            log_on_start = tuple(log_on_start)
        self.log_on_start = log_on_start

    def _on_training_start(self) -> None:

        _logger = self.globals["logger"].Logger.CURRENT
        _dir = _logger.dir
        if _dir is None:
            warnings.warn("Logging directory is missing. Skipping the logger {}".format(self._format))
            return None
        log_format = logger.make_output_format(self._format, _dir, self.suffix)
        _logger.output_formats.append(log_format)
        self.log_on_start = (("log_path", _dir), *self.log_on_start)
        if self.log_on_start is not None:
            for pair in self.log_on_start:
                _logger.record(*pair, ("tensorboard", "stdout"))

    def _on_step(self) -> bool:
        """
        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True
