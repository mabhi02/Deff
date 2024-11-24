import torch
from roboflow import Roboflow
import os
import yaml
import shutil
from pathlib import Path
import sys
from tqdm import tqdm
import logging
import traceback
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def setup_cuda() -> Tuple[bool, str]:
    """Configure CUDA settings and return device information."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda:0')
        device_name = torch.cuda.get_device_name(0)
        torch.backends.cudnn.benchmark = True
        logging.info(f"Using GPU: {device_name}")
        return True, device_name
    else:
        logging.warning("CUDA not available. Using CPU.")
        return False, "cpu"

def train_model(yaml_path: str) -> bool:
    """
    Train YOLOv5 model using the direct training script.
    
    Args:
        yaml_path (str): Path to the dataset YAML file
    Returns:
        bool: Whether training completed successfully
    """
    try:
        is_cuda_available, device_name = setup_cuda()
        
        # Import the training module from yolov5
        from yolov5 import train
        
        # Configure training arguments
        training_args = {
            'data': yaml_path,
            'epochs': 50,
            'weights': 'yolov5n.pt',
            'img': 640,
            'batch': 16 if is_cuda_available else 8,
            'device': 0 if is_cuda_available else 'cpu',
            'workers': 4 if is_cuda_available else 2,
            'project': 'runs/detect',
            'name': 'military_detection',
            'exist_ok': True,
            'patience': 20,
            'save_period': 10,
            'cache': True,
            'rect': True,
            'multi_scale': True,
            'optimizer': 'SGD',
            'sync_bn': True if is_cuda_available else False,
            'label_smoothing': 0.1,
            'freeze': [0, 1, 2, 3]  # Freeze first few layers during initial training
        }
        
        # Start training using the train module
        results = train.run(**training_args)
        
        logging.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False

def create_unified_dataset() -> Tuple[str, str]:
    """Create and prepare the unified dataset with progress tracking."""
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

    base_dir = "unified_military_dataset"
    os.makedirs(base_dir, exist_ok=True)

    # Download datasets with progress tracking
    datasets_to_download = [
        (rf.workspace("sputnik-yqqms").project("military-targets").version(1), "military-targets"),
        (rf.workspace("project-1jdii").project("valeriia-7cu35").version(1), "valeriia"),
        (rf.workspace("sputnik-yqqms").project("military-objects-9pdm9").version(1), "military-objects"),
        (rf.workspace("militaryvehiclerecognition").project("military-vehicle-recognition").version(1), "military-vehicle-recognition")
    ]

    downloaded_datasets = []
    with tqdm(total=len(datasets_to_download), desc="Downloading datasets") as pbar:
        for project, name in datasets_to_download:
            try:
                logging.info(f"Downloading {name} dataset...")
                dataset = project.download("yolov5")
                downloaded_datasets.append((dataset.location, name))
                pbar.update(1)
            except Exception as e:
                logging.error(f"Error downloading {name}: {e}")
                continue

    # Process datasets with progress tracking
    total_files = sum(len(os.listdir(os.path.join(path, split, 'images')))
                     for path, _ in downloaded_datasets
                     for split in ['train', 'valid', 'test']
                     if os.path.exists(os.path.join(path, split, 'images')))

    with tqdm(total=total_files, desc="Processing images") as pbar:
        for dataset_path, dataset_name in downloaded_datasets:
            for split in ['train', 'valid', 'test']:
                os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
                os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

                src_img_dir = os.path.join(dataset_path, split, 'images')
                src_label_dir = os.path.join(dataset_path, split, 'labels')
                
                if os.path.exists(src_img_dir):
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
                                                class_name = parts[0]
                                                new_class = class_mappings[dataset_name].get(class_name)
                                            else:
                                                old_class = int(parts[0])
                                                new_class = class_mappings[dataset_name].get(old_class)
                                            
                                            if new_class is not None:
                                                new_lines.append(f"{new_class} {' '.join(parts[1:])}\n")
                                    
                                    dst_label = os.path.join(base_dir, split, 'labels', f"{dataset_name}_{base_name}.txt")
                                    with open(dst_label, 'w') as f:
                                        f.writelines(new_lines)
                                except Exception as e:
                                    logging.error(f"Error processing label file {label_file}: {e}")
                            
                            pbar.update(1)

    return base_dir, unified_classes

def create_yaml_config(base_dir: str, unified_classes: dict) -> str:
    """Create the YAML configuration file for training."""
    data_yaml = {
        'path': os.path.abspath(base_dir),  # Full path to dataset directory
        'train': os.path.join('train', 'images'),  # Relative paths from dataset root
        'val': os.path.join('valid', 'images'),
        'test': os.path.join('test', 'images'),
        
        # Class configurations
        'nc': len(unified_classes),  # Number of classes
        'names': list(unified_classes.values()),  # Class names
        
        # Additional configurations
        'download': False,  # Don't download dataset
        'cache': True,  # Cache images for faster training
    }
    
    yaml_path = os.path.join(base_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    
    return yaml_path

if __name__ == "__main__":
    try:
        # Clean up previous training artifacts
        if os.path.exists('runs/detect/military_detection'):
            shutil.rmtree('runs/detect/military_detection')
        
        logging.info("Creating unified dataset...")
        base_dir, unified_classes = create_unified_dataset()
        
        # Ensure YAML file is properly configured
        yaml_path = create_yaml_config(base_dir, unified_classes)
        
        logging.info("Starting model training...")
        train_success = train_model(yaml_path)
        
        if train_success:
            # Save the best model
            best_weights_path = 'runs/detect/military_detection/weights/best.pt'
            if os.path.exists(best_weights_path):
                final_path = os.path.join(base_dir, 'best.pt')
                shutil.copy(best_weights_path, final_path)
                logging.info(f"Best model saved to {final_path}")
            else:
                logging.warning("Best weights file not found at expected location")
        else:
            logging.error("Training failed! Check training.log for details.")
            
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)