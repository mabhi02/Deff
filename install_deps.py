import os
import sys
import subprocess

def install_dependencies():
    # List of packages to uninstall first
    to_uninstall = [
        'numpy',
        'ultralytics',
        'torch',
        'torchvision',
        'yolov5'
    ]
    
    # Uninstall existing packages
    for package in to_uninstall:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', package])
        except:
            pass
    
    # Install packages in correct order
    install_commands = [
        [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
        [sys.executable, '-m', 'pip', 'install', 'numpy==1.24.3'],  # Specific version for compatibility
        [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu118'],
        [sys.executable, '-m', 'pip', 'install', 'opencv-python-headless'],
        [sys.executable, '-m', 'pip', 'install', 'yolov5']
    ]
    
    for cmd in install_commands:
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Error installing with command {' '.join(cmd)}: {e}")
            sys.exit(1)
    
    print("All dependencies installed successfully!")

if __name__ == "__main__":
    install_dependencies()