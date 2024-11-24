import torch
from roboflow import Roboflow
import os
import yaml
import shutil
from pathlib import Path
import subprocess
import sys

def train_model(yaml_path):
    if torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    # Direct training command using yolov5 train
    cmd = [
        "yolov5", "train",
        "--data", yaml_path,
        "--img", "640",
        "--batch", "16",
        "--epochs", "100",
        "--weights", "yolov5n.pt",
        "--project", "runs/detect",
        "--name", "military_detection",
        "--cache",
        "--device", "0",
        "--workers", "4",
        "--patience", "20",
        "--save-period", "10",
        "--exist-ok",
        "--rect"
    ]
    
    try:
        # Create a log file for output
        log_file = "training_log.txt"
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Print output in real-time
            for line in process.stdout:
                print(line, end='')
                f.write(line)
            
            process.wait()
            if process.returncode == 0:
                return True
            else:
                print(f"Training failed with exit code {process.returncode}")
                return False
                
    except Exception as e:
        print(f"Training failed with error: {e}")
        return False

def create_unified_dataset():
    rf = Roboflow(api_key="orJDGfanyY3nTxu3YQKZ")
    
    unified_classes = {
        0: 'military_vehicle',
        1: 'aircraft',
        2: 'soldier',
        3: 'civilian',
        4: 'ordnance'
    }
    
    class_mappings = {
        'military-targets': {
            0: 0,  # car -> military_vehicle
            1: 1,  # drone_plane -> aircraft
            2: 1,  # drone_quadro -> aircraft
            3: 1,  # helicopter -> aircraft
            4: 0,  # military_car -> military_vehicle
            5: 4,  # ordnance -> ordnance
            6: 2,  # soldier -> soldier
            7: 0,  # tank -> military_vehicle
        },
        'valeriia': {
            0: 3,  # civilian -> civilian
            1: 1,  # fpv -> aircraft
            2: 2,  # soldier -> soldier
            3: 2,  # soldier_UA -> soldier
            4: 2,  # soldier_ru -> soldier
            5: 2,  # soldier_ua -> soldier
        },
        'military-objects': {
            0: 0,  # military vehicle -> military_vehicle
            1: 1,  # aircraft -> aircraft
        },
        'military-vehicle-recognition': {
            'air-fighter': 1,       # aircraft
            'armoured personnel carrier': 0,  # military_vehicle
            'bomber': 1,           # aircraft
            'soldier': 2,          # soldier
            'tank': 0             # military_vehicle
        }
    }

    # Create base directory
    base_dir = "unified_military_dataset"  # Simplified path
    os.makedirs(base_dir, exist_ok=True)

    # Download datasets
    print("Downloading datasets...")
    datasets_to_download = [
        (rf.workspace("sputnik-yqqms").project("military-targets").version(1), "military-targets"),
        (rf.workspace("project-1jdii").project("valeriia-7cu35").version(1), "valeriia"),
        (rf.workspace("sputnik-yqqms").project("military-objects-9pdm9").version(1), "military-objects"),
        (rf.workspace("militaryvehiclerecognition").project("military-vehicle-recognition").version(1), "military-vehicle-recognition")
    ]

    downloaded_datasets = []
    for project, name in datasets_to_download:
        try:
            print(f"Downloading {name} dataset...")
            dataset = project.download("yolov5")
            downloaded_datasets.append((dataset.location, name))
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            continue

    # Process datasets
    for dataset_path, dataset_name in downloaded_datasets:
        for split in ['train', 'valid', 'test']:
            # Create directories if they don't exist
            os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

            src_img_dir = os.path.join(dataset_path, split, 'images')
            src_label_dir = os.path.join(dataset_path, split, 'labels')
            
            if os.path.exists(src_img_dir):
                print(f"Processing {dataset_name} {split} set...")
                for img_file in os.listdir(src_img_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        base_name = os.path.splitext(img_file)[0]
                        
                        # Copy image
                        src_img = os.path.join(src_img_dir, img_file)
                        dst_img = os.path.join(base_dir, split, 'images', f"{dataset_name}_{img_file}")
                        shutil.copy2(src_img, dst_img)
                        
                        # Process label
                        label_file = os.path.join(src_label_dir, f"{base_name}.txt")
                        if os.path.exists(label_file):
                            try:
                                with open(label_file, 'r') as f:
                                    lines = f.readlines()
                                
                                new_lines = []
                                for line in lines:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        if dataset_name == 'military-vehicle-recognition':
                                            # Handle string class names
                                            class_name = parts[0]
                                            new_class = class_mappings[dataset_name].get(class_name)
                                        else:
                                            # Handle numeric class indices
                                            old_class = int(parts[0])
                                            new_class = class_mappings[dataset_name].get(old_class)
                                        
                                        if new_class is not None:
                                            new_lines.append(f"{new_class} {' '.join(parts[1:])}\n")
                                
                                dst_label = os.path.join(base_dir, split, 'labels', f"{dataset_name}_{base_name}.txt")
                                with open(dst_label, 'w') as f:
                                    f.writelines(new_lines)
                            except Exception as e:
                                print(f"Error processing label file {label_file}: {e}")

    # Create data.yaml with simplified paths
    data_yaml = {
        'path': os.path.abspath(base_dir),  # Full path to dataset directory
        'train': 'train/images',  # Relative paths from dataset root
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(unified_classes),
        'names': list(unified_classes.values())
    }
    
    yaml_path = os.path.join(base_dir, 'data.yaml')
    print(f"\nWriting data configuration to {yaml_path}")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # Print dataset statistics
    print("\nDataset Statistics:")
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(base_dir, split, 'images')
        if os.path.exists(img_dir):
            num_images = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{split}: {num_images} images")

    return base_dir, yaml_path

if __name__ == "__main__":
    # Clean up any previous training artifacts
    if os.path.exists('runs/detect/military_detection'):
        shutil.rmtree('runs/detect/military_detection')
    
    print("Creating unified dataset...")
    base_dir, yaml_path = create_unified_dataset()
    
    print("Starting model training...")
    train_success = train_model(yaml_path)
    
    if train_success:
        print("\nTraining completed successfully!")
        
        # Save the best model
        best_weights_path = 'runs/detect/military_detection/weights/best.pt'
        if os.path.exists(best_weights_path):
            final_path = os.path.join(base_dir, 'best.pt')
            shutil.copy(best_weights_path, final_path)
            print(f"Best model saved to {final_path}")
    else:
        print("\nTraining failed! Check training_log.txt for details.")