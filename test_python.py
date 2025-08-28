import sys
import os

print("Python Environment Test")
print("=====================")
print(f"Python Version: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Working Directory: {os.getcwd()}")
print("\nEnvironment Variables:")
for key, value in os.environ.items():
    if 'python' in key.lower() or 'path' in key.lower():
        print(f"{key}: {value}")

print("\nBasic Math Test:")
print(f"2 + 2 = {2+2}")
print("\nTest completed successfully!")
