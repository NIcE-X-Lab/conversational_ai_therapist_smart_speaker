import Jetson.GPIO as GPIO
import sys

def find_free_pins():
    GPIO.setmode(GPIO.BOARD)
    all_pins = [7, 11, 12, 13, 15, 16, 18, 19, 21, 22, 23, 24, 26, 29, 31, 32, 33, 35, 36, 37, 38, 40]
    
    print("--- Available Input Pins ---")
    for pin in all_pins:
        try:
            GPIO.setup(pin, GPIO.IN)
            val = GPIO.input(pin)
            print(f"Pin {pin:02d}: {'HIGH' if val else 'LOW'}")
        except Exception as e:
            print(f"Pin {pin:02d}: BUSY ({str(e).split(':')[-1].strip()})")
    GPIO.cleanup()

if __name__ == "__main__":
    find_free_pins()
