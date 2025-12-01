#!/usr/bin/env python3
"""
MGB Data Loading Helper

This script helps load MGB data for pathway validation analysis.
You'll need to specify the correct paths to your MGB data files.
"""

import torch
import numpy as np
import pandas as pd
import os

def load_mgb_data():
    """
    Load MGB dataset for pathway validation
    
    You'll need to specify the correct paths to your MGB data files:
    - MGB Y matrix (binary disease events)
    - MGB thetas (signature loadings) 
    - MGB disease names
    - MGB patient IDs (from your R output)
    - MGB medication data (from your R output)
    """
    
    print("Loading MGB dataset...")
    
    # TODO: Specify correct paths to your MGB data files
    mgb_data_paths = {
        'Y_matrix': '/path/to/mgb/Y_tensor.pt',  # Binary disease events
        'thetas': '/path/to/mgb/thetas.npy',     # Signature loadings
        'disease_names': '/path/to/mgb/disease_names.csv',  # Disease names
        'patient_ids': '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/mgbnames.csv',  # From your R output
        'medications': '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/medsformgbtopic.csv'  # From your R output
    }
    
    # Load patient IDs (you already have this)
    if os.path.exists(mgb_data_paths['patient_ids']):
        mgb_names = pd.read_csv(mgb_data_paths['patient_ids'])
        mgb_ids = mgb_names.iloc[:, 0].values  # First column contains the IDs
        print(f"✅ Loaded {len(mgb_ids)} MGB patient IDs")
    else:
        print(f"❌ Could not find MGB patient IDs at {mgb_data_paths['patient_ids']}")
        mgb_ids = None
    
    # Load medication data (you already have this)
    if os.path.exists(mgb_data_paths['medications']):
        mgb_meds = pd.read_csv(mgb_data_paths['medications'])
        print(f"✅ Loaded MGB medication data: {mgb_meds.shape}")
    else:
        print(f"❌ Could not find MGB medication data at {mgb_data_paths['medications']}")
        mgb_meds = None
    
    # Load Y matrix (you need to specify the correct path)
    if os.path.exists(mgb_data_paths['Y_matrix']):
        Y_mgb = torch.load(mgb_data_paths['Y_matrix'])
        print(f"✅ Loaded MGB Y matrix: {Y_mgb.shape}")
    else:
        print(f"❌ Could not find MGB Y matrix at {mgb_data_paths['Y_matrix']}")
        print("   Please specify the correct path to your MGB Y_tensor.pt file")
        Y_mgb = None
    
    # Load thetas (you need to specify the correct path)
    if os.path.exists(mgb_data_paths['thetas']):
        thetas_mgb = np.load(mgb_data_paths['thetas'])
        print(f"✅ Loaded MGB thetas: {thetas_mgb.shape}")
    else:
        print(f"❌ Could not find MGB thetas at {mgb_data_paths['thetas']}")
        print("   Please specify the correct path to your MGB thetas.npy file")
        thetas_mgb = None
    
    # Load disease names (you need to specify the correct path)
    if os.path.exists(mgb_data_paths['disease_names']):
        disease_names_df = pd.read_csv(mgb_data_paths['disease_names'])
        disease_names_mgb = disease_names_df.iloc[:, 0].tolist()  # First column
        print(f"✅ Loaded {len(disease_names_mgb)} MGB disease names")
    else:
        print(f"❌ Could not find MGB disease names at {mgb_data_paths['disease_names']}")
        print("   Please specify the correct path to your MGB disease_names.csv file")
        disease_names_mgb = None
    
    return Y_mgb, thetas_mgb, disease_names_mgb, mgb_ids, mgb_meds

def check_mgb_data_availability():
    """
    Check what MGB data files are available
    """
    print("Checking MGB data availability...")
    
    # Check for common MGB data file locations
    possible_locations = [
        '/Users/sarahurbut/aladynoulli2/pyScripts/mgb/',
        '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/mgb_data/',
        '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/mgb/',
        '/Users/sarahurbut/aladynoulli2/mgb_data/',
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            print(f"✅ Found directory: {location}")
            files = os.listdir(location)
            print(f"   Files: {files[:10]}...")  # Show first 10 files
        else:
            print(f"❌ Directory not found: {location}")
    
    # Check for the files you mentioned from R
    r_output_files = [
        '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/mgbnames.csv',
        '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/medsformgbtopic.csv'
    ]
    
    print("\nChecking R output files:")
    for file_path in r_output_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
            df = pd.read_csv(file_path)
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
        else:
            print(f"❌ Not found: {file_path}")

def create_mgb_data_template():
    """
    Create a template for MGB data loading
    """
    template = """
# MGB Data Loading Template
# Replace the paths below with your actual MGB data file locations

def load_mgb_data():
    # MGB Y matrix (binary disease events: patients x diseases x time)
    Y_mgb = torch.load('/path/to/your/mgb/Y_tensor.pt')
    
    # MGB thetas (signature loadings: patients x signatures x time)  
    thetas_mgb = np.load('/path/to/your/mgb/thetas.npy')
    
    # MGB disease names
    disease_names_df = pd.read_csv('/path/to/your/mgb/disease_names.csv')
    disease_names_mgb = disease_names_df.iloc[:, 0].tolist()
    
    # MGB patient IDs (from your R output)
    mgb_names = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/mgbnames.csv')
    mgb_ids = mgb_names.iloc[:, 0].values
    
    # MGB medication data (from your R output)
    mgb_meds = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/medsformgbtopic.csv')
    
    return Y_mgb, thetas_mgb, disease_names_mgb, mgb_ids, mgb_meds
"""
    
    with open('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/mgb_data_template.py', 'w') as f:
        f.write(template)
    
    print("Created MGB data template at: mgb_data_template.py")

if __name__ == "__main__":
    print("MGB Data Loading Helper")
    print("="*50)
    
    # Check what data is available
    check_mgb_data_availability()
    
    # Create template
    create_mgb_data_template()
    
    # Try to load what we can
    print("\nAttempting to load available MGB data...")
    Y_mgb, thetas_mgb, disease_names_mgb, mgb_ids, mgb_meds = load_mgb_data()
    
    if mgb_ids is not None and mgb_meds is not None:
        print("\n✅ Successfully loaded MGB patient IDs and medication data")
        print("   You can now run pathway analysis once you specify the correct paths")
        print("   for Y_mgb, thetas_mgb, and disease_names_mgb")
    else:
        print("\n❌ Could not load MGB data - please check file paths")
