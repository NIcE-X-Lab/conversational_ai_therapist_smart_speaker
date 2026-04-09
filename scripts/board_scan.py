#!/usr/bin/env python3
import Jetson.GPIO as GPIO
import time
import sys

def scan():
    GPIO.setmode(GPIO.BOARD)
    # List of common 40-pin header pins that are likely GPIOs
    # Excluding GND, 3.3V, 5V
    pins = [7, 11, 12, 13, 15, 16, 18, 19, 21, 22, 23, 24, 26, 29, 31, 32, 33, 35, 36, 37, 38, 40]
    
    print("--- CaiTI Board Mapping Scanner ---")
    print("This will toggle each pin for 2 seconds. Watch the LED!")
    
    try:
        for pin in pins:
            print(f"Testing Board Pin {pin}...", end="", flush=True)
            try:
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
                GPIO.output(pin, GPIO.HIGH)
                time.sleep(2)
                GPIO.output(pin, GPIO.LOW)
                GPIO.cleanup(pin)
                print(" Done.")
            except Exception as e:
                print(f" Skipped ({e})")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nScan interrupted.")
    finally:
        GPIO.cleanup()
        print("Scan finished.")

if __name__ == "__main__":
    scan()
