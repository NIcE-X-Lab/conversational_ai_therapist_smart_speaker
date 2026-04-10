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
import time
from src.utils.log_util import get_logger
from src.utils.config_loader import (
    PIN_BTN_START,
    PIN_BTN_END,
    PIN_BTN_OPT_OUT,
    PIN_BTN_4,
    PIN_LISTENING_LED,
    PIN_LISTENING_LED_ACTIVE_LOW,
    PIN_BUTTONS_ACTIVE_LOW,
    PIN_BTN_START_ACTIVE_LOW,
    PIN_BTN_END_ACTIVE_LOW,
    PIN_BTN_OPT_OUT_ACTIVE_LOW,
    PIN_BTN_4_ACTIVE_LOW,
)

logger = get_logger("GPIOManager")

# GPIO pin constants (BOARD numbering) mapped from config
PIN_START_SESSION = PIN_BTN_START
PIN_END_SESSION   = PIN_BTN_END
PIN_OPT_OUT       = PIN_BTN_OPT_OUT
PIN_BTN4          = PIN_BTN_4
PIN_LED_LISTEN    = PIN_LISTENING_LED

# Event types pushed onto the shared queue
EVENT_START   = "START_SESSION"
EVENT_END     = "END_SESSION"
EVENT_OPT_OUT = "OPT_OUT"
EVENT_BTN4    = "BTN4"

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
    RISING = "RISING"
    BOTH = "BOTH"
    PUD_UP  = "PUD_UP"
    PUD_DOWN = "PUD_DOWN"

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
        self._debounce_sec = 0.30

        # Step 1: Physical Setup FIRST
        self._setup_gpio()

        # Step 2: Configure state tracking with the now-configured hardware
        self._active_low_by_pin = {
            PIN_START_SESSION: PIN_BTN_START_ACTIVE_LOW,
            PIN_END_SESSION: PIN_BTN_END_ACTIVE_LOW,
            PIN_OPT_OUT: PIN_BTN_OPT_OUT_ACTIVE_LOW,
            PIN_BTN4: PIN_BTN_4_ACTIVE_LOW,
        }
        self._last_emit = {
            EVENT_START: 0.0,
            EVENT_END: 0.0,
            EVENT_OPT_OUT: 0.0,
            EVENT_BTN4: 0.0,
        }
        self._last_state = {
            PIN_START_SESSION: self._safe_input(PIN_START_SESSION),
            PIN_END_SESSION: self._safe_input(PIN_END_SESSION),
            PIN_OPT_OUT: self._safe_input(PIN_OPT_OUT),
            PIN_BTN4: self._safe_input(PIN_BTN4),
        }

    # ------------------------------------------------------------------ #
    # Setup                                                                 #
    # ------------------------------------------------------------------ #

    def _setup_gpio(self):
        try:
            _gpio.setmode(_gpio.BOARD)

            # Buttons mapping
            btn_configs = [
                (PIN_START_SESSION, PIN_BTN_START_ACTIVE_LOW, self._on_start),
                (PIN_END_SESSION, PIN_BTN_END_ACTIVE_LOW, self._on_end),
                (PIN_OPT_OUT, PIN_BTN_OPT_OUT_ACTIVE_LOW, self._on_opt_out),
                (PIN_BTN4, PIN_BTN_4_ACTIVE_LOW, self._on_btn4),
            ]

            for pin, active_low, callback in btn_configs:
                # If active-low (connect to GND), we need a pull-UP to hold it HIGH when open.
                # If active-high (connect to VCC), we need a pull-DOWN to hold it LOW when open.
                pull = _gpio.PUD_UP if active_low else _gpio.PUD_DOWN
                _gpio.setup(pin, _gpio.IN, pull_up_down=pull)

                # Detection (FALLING for active-low, RISING for active-high)
                edge = _gpio.FALLING if active_low else _gpio.RISING
                _gpio.add_event_detect(pin, edge, callback=callback, bouncetime=300)

            # LED — output, default OFF
            _gpio.setup(PIN_LED_LISTEN, _gpio.OUT)
            self._write_led(False)

            logger.info(f"GPIO configured. Listening LED: {PIN_LED_LISTEN}, Buttons: {[p for p,_,_ in btn_configs]}")
        except Exception as e:
            logger.error(f"GPIO setup critical failure: {e}")

    def _safe_input(self, pin: int) -> int:
        try:
            return int(_gpio.input(pin))
        except Exception:
            return 1 if PIN_BUTTONS_ACTIVE_LOW else 0

    def _is_pressed(self, pin: int) -> bool:
        level = self._safe_input(pin)
        active_low = self._active_low_by_pin.get(pin, PIN_BUTTONS_ACTIVE_LOW)
        return (level == 0) if active_low else (level == 1)

    def _emit_event(self, event_name: str, channel: int, source: str = "GPIO"):
        now = time.monotonic()
        if now - self._last_emit[event_name] < self._debounce_sec:
            return
        self._last_emit[event_name] = now
        logger.info(f"[{source}] {event_name} pressed (pin {channel})")
        self.gpio_event_queue.put(event_name)
        self.flash_led(0.08)

    def _write_led(self, state: bool):
        # state=True means LED should be visibly ON
        level = (not state) if PIN_LISTENING_LED_ACTIVE_LOW else state
        try:
            _gpio.output(PIN_LED_LISTEN, level)
            logger.debug(f"LED (Pin {PIN_LED_LISTEN}) set to {level} (requested ON={state})")
        except Exception as e:
            logger.debug(f"LED write failed: {e}")

    # ------------------------------------------------------------------ #
    # Callbacks (run in GPIO interrupt thread)                             #
    # ------------------------------------------------------------------ #

    def _on_start(self, channel):
        try:
            if self._is_pressed(channel):
                self._emit_event(EVENT_START, channel, source="GPIO")
        except Exception as e:
            logger.error(f"GPIO callback _on_start failed: {e}")

    def _on_end(self, channel):
        try:
            if self._is_pressed(channel):
                self._emit_event(EVENT_END, channel, source="GPIO")
        except Exception as e:
            logger.error(f"GPIO callback _on_end failed: {e}")

    def _on_opt_out(self, channel):
        try:
            if self._is_pressed(channel):
                self._emit_event(EVENT_OPT_OUT, channel, source="GPIO")
        except Exception as e:
            logger.error(f"GPIO callback _on_opt_out failed: {e}")

    def _on_btn4(self, channel):
        try:
            if self._is_pressed(channel):
                self._emit_event(EVENT_BTN4, channel, source="GPIO")
        except Exception as e:
            logger.error(f"GPIO callback _on_btn4 failed: {e}")

    # ------------------------------------------------------------------ #
    # LED control                                                          #
    # ------------------------------------------------------------------ #

    def set_led(self, state: bool):
        """Turn the listening indicator LED on (True) or off (False)."""
        try:
            self._write_led(state)
        except Exception as e:
            logger.debug(f"LED set failed (stub?): {e}")

    def flash_led(self, duration_sec: float = 0.10):
        def _blink():
            try:
                self.set_led(True)
                time.sleep(duration_sec)
                self.set_led(False)
            except Exception:
                pass
        threading.Thread(target=_blink, daemon=True).start()

    def poll_event(self):
        """Non-blocking event read with fallback level polling."""
        try:
            return self.gpio_event_queue.get_nowait()
        except queue.Empty:
            pass

        self._poll_buttons_fallback()
        try:
            return self.gpio_event_queue.get_nowait()
        except queue.Empty:
            return None

    def _poll_buttons_fallback(self):
        pin_to_event = {
            PIN_START_SESSION: EVENT_START,
            PIN_END_SESSION: EVENT_END,
            PIN_OPT_OUT: EVENT_OPT_OUT,
            PIN_BTN4: EVENT_BTN4,
        }
        for pin, event_name in pin_to_event.items():
            cur = self._safe_input(pin)
            prev = self._last_state.get(pin, cur)
            prev_pressed = ((prev == 0) if self._active_low_by_pin.get(pin, PIN_BUTTONS_ACTIVE_LOW) else (prev == 1))
            cur_pressed = ((cur == 0) if self._active_low_by_pin.get(pin, PIN_BUTTONS_ACTIVE_LOW) else (cur == 1))
            if cur != prev:
                self._last_state[pin] = cur
            if cur_pressed and not prev_pressed:
                self._emit_event(event_name, pin, source="GPIO-POLL")

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #

    def cleanup(self):
        try:
            self.set_led(False)
            for pin in (PIN_START_SESSION, PIN_END_SESSION, PIN_OPT_OUT, PIN_BTN4):
                _gpio.remove_event_detect(pin)
            _gpio.cleanup()
            logger.info("GPIO cleaned up.")
        except Exception as e:
            logger.debug(f"GPIO cleanup error (stub?): {e}")
