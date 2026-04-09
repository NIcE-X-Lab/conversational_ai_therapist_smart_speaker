#!/usr/bin/env python3
"""
scripts/gpio_probe.py — Hardware Diagnostic Tool for CaiTI.
Toggles the listening LED and prints button states to verify wiring.
"""
import sys
import os
import time

# Add root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drivers.gpio_manager import GPIOManager, EVENT_START, EVENT_END, EVENT_OPT_OUT, EVENT_BTN4
from src.utils.config_loader import (
    PIN_BTN_START, PIN_BTN_END, PIN_BTN_OPT_OUT, PIN_BTN_4, PIN_LISTENING_LED
)

def run_probe():
    print("--- CaiTI GPIO Probe Tool ---")
    print(f"Config: LED={PIN_LISTENING_LED}, START={PIN_BTN_START}, END={PIN_BTN_END}, OPT={PIN_BTN_OPT_OUT}, BTN4={PIN_BTN_4}")
    
    try:
        gpio = GPIOManager()
    except Exception as e:
        print(f"FAILED to initialize GPIOManager: {e}")
        sys.exit(1)

    print("\n[Diagnostic 1] Testing LED...")
    for i in range(3):
        print(f"  LED ON... (Iteration {i+1}/3)")
        gpio.set_led(True)
        time.sleep(1)
        print("  LED OFF...")
        gpio.set_led(False)
        time.sleep(0.5)

    print("\n[Diagnostic 2] Monitoring Buttons (Ctrl+C to stop)...")
    print("Press your physical buttons now to see if they register.")
    
    try:
        while True:
            # Check for events in the queue (interrupt-driven)
            event = gpio.poll_event()
            if event:
                print(f"  >>> EVENT DETECTED: {event}")
            
            # Periodically print raw levels for debugging
            # Note: access private _is_pressed for probe purposes
            start_p = gpio._is_pressed(PIN_BTN_START)
            end_p = gpio._is_pressed(PIN_BTN_END)
            opt_p = gpio._is_pressed(PIN_BTN_OPT_OUT)
            btn4_p = gpio._is_pressed(PIN_BTN_4)
            
            # Only print if something is pressed to reduce spam
            if start_p or end_p or opt_p or btn4_p:
                print(f"  [Raw State] START:{start_p} END:{end_p} OPT:{opt_p} BTN4:{btn4_p}")
                time.sleep(0.2)
            
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping probe...")
    finally:
        gpio.cleanup()
        print("GPIO cleaned up.")

if __name__ == "__main__":
    run_probe()
