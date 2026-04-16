# AirInk

A real-time drawing app that uses your webcam to track a colored object (like a green bottle cap) or your finger and paints its trail across the screen.

What it does

Captures live video from your default camera.

Finds the largest contour of that color and treats its center as the "brush tip".

Draws continuous lines between consecutive tip positions, leaving a colored trail on a persistent canvas overlay.

Lets you change brush color and thickness on the fly using the keyboard.

Press c to wipe the canvas clean, s to save your current drawing as a PNG.

Requirements
Python 3.12+

OpenCV (cv2)

NumPy
License
MIT License.