"""Low-level driver handling Jetson GPIO physical pins and LEDs."""
"""
gpio_manager.py — Jetson GPIO / Physical Hardware Interface for CaiTI.

Pin Map (BOARD numbering, active-low buttons):
  Pin 11 → START_SESSION  (FALLING edge, 300ms debounce)
  Pin 13 → END_SESSION    (FALLING edge, 300ms debounce)
  Pin 15 → OPT_OUT        (FALLING edge, 300ms debounce)
  Pin 18 → LISTENING_LED  (OUTPUT — HIGH while Whisper is recording)

Graceful degradation: if Jetson.GPIO is unavailable (dev machine),
all methods become no-ops and a single warning is emitted.
"""

import queue
import threading
from src.utils.log_util import get_logger
from src.utils.config_loader import (
    PIN_BTN_START,
    PIN_BTN_END,
    PIN_BTN_OPT_OUT,
    PIN_LISTENING_LED
)

logger = get_logger("GPIOManager")

# GPIO pin constants (BOARD numbering) mapped from config
PIN_START_SESSION = PIN_BTN_START
PIN_END_SESSION   = PIN_BTN_END
PIN_OPT_OUT       = PIN_BTN_OPT_OUT
PIN_LED_LISTEN    = PIN_LISTENING_LED

# Event types pushed onto the shared queue
EVENT_START   = "START_SESSION"
EVENT_END     = "END_SESSION"
EVENT_OPT_OUT = "OPT_OUT"

# Try importing Jetson.GPIO; fall back to stub if not on Jetson
try:
    import Jetson.GPIO as GPIO
    _GPIO_AVAILABLE = True
    logger.info("Jetson.GPIO loaded — physical hardware active.")
except ImportError:
    _GPIO_AVAILABLE = False
    logger.warning("Jetson.GPIO not available — running in stub/no-op mode.")


class _GPIOStub:
    """No-op stub used on non-Jetson hardware."""
    BOARD = "BOARD"
    OUT   = "OUT"
    IN    = "IN"
    FALLING = "FALLING"
    PUD_UP  = "PUD_UP"

    def setmode(self, *a, **kw): pass
    def setup(self, *a, **kw): pass
    def output(self, *a, **kw): pass
    def input(self, pin): return 1          # unpressed (active-low)
    def add_event_detect(self, *a, **kw): pass
    def remove_event_detect(self, *a, **kw): pass
    def cleanup(self, *a, **kw): pass


_gpio = GPIO if _GPIO_AVAILABLE else _GPIOStub()


class GPIOManager:
    """
    Singleton-style GPIO manager.
    Call gpio_event_queue.get_nowait() in your main loop
    to consume button events without blocking.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.gpio_event_queue: queue.Queue = queue.Queue()
        self._setup_gpio()

    # ------------------------------------------------------------------ #
    # Setup                                                                 #
    # ------------------------------------------------------------------ #

    def _setup_gpio(self):
        try:
            _gpio.setmode(_gpio.BOARD)

            # Buttons — input with internal pull-up (active-low)
            for pin in (PIN_START_SESSION, PIN_END_SESSION, PIN_OPT_OUT):
                _gpio.setup(pin, _gpio.IN, pull_up_down=_gpio.PUD_UP)

            # LED — output, default OFF
            _gpio.setup(PIN_LED_LISTEN, _gpio.OUT)
            _gpio.output(PIN_LED_LISTEN, False)

            # Edge detection with 300 ms debounce
            _gpio.add_event_detect(
                PIN_START_SESSION, _gpio.FALLING,
                callback=self._on_start,  bouncetime=300
            )
            _gpio.add_event_detect(
                PIN_END_SESSION,   _gpio.FALLING,
                callback=self._on_end,    bouncetime=300
            )
            _gpio.add_event_detect(
                PIN_OPT_OUT,       _gpio.FALLING,
                callback=self._on_opt_out, bouncetime=300
            )
            logger.info("GPIO pins configured: 11=START, 13=END, 15=OPT_OUT, 18=LED.")
        except Exception as e:
            logger.error(f"GPIO setup failed: {e}")

    # ------------------------------------------------------------------ #
    # Callbacks (run in GPIO interrupt thread)                             #
    # ------------------------------------------------------------------ #

    def _on_start(self, channel):
        logger.info(f"[GPIO] START_SESSION pressed (pin {channel})")
        self.gpio_event_queue.put(EVENT_START)

    def _on_end(self, channel):
        logger.info(f"[GPIO] END_SESSION pressed (pin {channel})")
        self.gpio_event_queue.put(EVENT_END)

    def _on_opt_out(self, channel):
        logger.info(f"[GPIO] OPT_OUT pressed (pin {channel})")
        self.gpio_event_queue.put(EVENT_OPT_OUT)

    # ------------------------------------------------------------------ #
    # LED control                                                          #
    # ------------------------------------------------------------------ #

    def set_led(self, state: bool):
        """Turn the listening indicator LED on (True) or off (False)."""
        try:
            _gpio.output(PIN_LED_LISTEN, state)
        except Exception as e:
            logger.debug(f"LED set failed (stub?): {e}")

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #

    def cleanup(self):
        try:
            self.set_led(False)
            for pin in (PIN_START_SESSION, PIN_END_SESSION, PIN_OPT_OUT):
                _gpio.remove_event_detect(pin)
            _gpio.cleanup()
            logger.info("GPIO cleaned up.")
        except Exception as e:
            logger.debug(f"GPIO cleanup error (stub?): {e}")
