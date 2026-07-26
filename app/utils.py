"""
Utility functions for SMILES validation, descriptor generation, 
compound name lookup, and model inference.
"""

import os
import json
import requests
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from rdkit import Chem
from rdkit.Chem import Descriptors
import streamlit as st


# ============================================================================
# MODEL ARTIFACTS LOADING
# ============================================================================

@st.cache_resource
def load_model_artifacts() -> Tuple:
    """
    Load all model artifacts from the model/ directory.
    
    Returns:
        Tuple containing: (model, label_encoder, feature_columns, metadata)
    """
    try:
        model_path = "model/xgboost_model.joblib"
        encoder_path = "model/label_encoder.joblib"
        features_path = "model/feature_columns.joblib"
        metadata_path = "model/metadata.json"
        
        # Verify all files exist
        for path in [model_path, encoder_path, features_path, metadata_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
        
        # Load artifacts
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        feature_columns = joblib.load(features_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, label_encoder, feature_columns, metadata
    
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        raise


# ============================================================================
# SMILES VALIDATION
# ============================================================================

def validate_smiles(smiles: str) -> Tuple[bool, Optional[Chem.Mol]]:
    """
    Validate a SMILES string using RDKit.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        Tuple of (is_valid, mol_object)
    """
    if not smiles or not smiles.strip():
        return False, None
    
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return False, None
        return True, mol
    except Exception:
        return False, None


# ============================================================================
# DESCRIPTOR GENERATION
# ============================================================================

def generate_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Generate all RDKit descriptors for a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary of descriptor names and values
    """
    descriptors = {}
    
    for desc_name, desc_func in Descriptors.descList:
        try:
            value = desc_func(mol)
            # Handle NaN and infinite values
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            descriptors[desc_name] = float(value)
        except Exception:
            # If descriptor calculation fails, use 0.0
            descriptors[desc_name] = 0.0
    
    return descriptors


def prepare_features(
    descriptors: Dict[str, float], 
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Filter and reorder descriptors to match model's expected features.
    
    Args:
        descriptors: Dictionary of all generated descriptors
        feature_columns: List of feature names expected by the model
        
    Returns:
        DataFrame with features in correct order
    """
    # Extract only the features the model expects
    filtered_features = {}
    for feat in feature_columns:
        if feat in descriptors:
            filtered_features[feat] = descriptors[feat]
        else:
            # If feature is missing, use 0.0 as fallback
            filtered_features[feat] = 0.0
    
    # Create DataFrame with exact column order
    df = pd.DataFrame([filtered_features], columns=feature_columns)
    
    return df


# ============================================================================
# MODEL INFERENCE
# ============================================================================

def predict_toxicity(
    model, 
    features_df: pd.DataFrame, 
    label_encoder: Dict[int, str]
) -> Tuple[str, float, np.ndarray]:
    """
    Predict toxicity class and probabilities.
    
    Args:
        model: Trained XGBoost model
        features_df: DataFrame with features in correct order
        label_encoder: Dictionary mapping class indices to labels
        
    Returns:
        Tuple of (predicted_class, confidence, probabilities)
    """
    # Get prediction and probabilities
    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    
    # Get predicted class label
    predicted_class = label_encoder.get(int(prediction), f"Class {prediction}")
    
    # Get confidence (max probability)
    confidence = float(np.max(probabilities))
    
    return predicted_class, confidence, probabilities


# ============================================================================
# PUBCHEM COMPOUND LOOKUP
# ============================================================================

def get_compound_info(smiles: str, timeout: int = 5) -> dict:
    """
    Fetch comprehensive compound information from PubChem.
    
    Args:
        smiles: SMILES string of the compound
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with compound name, CID, formula, MW, and synonyms
    """
    info = {
        'name': None,
        'cid': None,
        'formula': None,
        'mw': None,
        'synonyms': [],
        'source': None
    }
    
    try:
        # Step 1: Get CID from SMILES
        url_cid = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
        response = requests.get(url_cid, timeout=timeout)
        
        if response.status_code != 200:
            return info
            
        cid_data = response.json()
        if 'IdentifierList' not in cid_data or 'CID' not in cid_data['IdentifierList']:
            return info
            
        cid = cid_data['IdentifierList']['CID'][0]
        info['cid'] = cid
        
        # Step 2: Get properties (IUPAC name, formula, MW)
        url_props = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName,MolecularFormula,MolecularWeight/JSON"
        response = requests.get(url_props, timeout=timeout)
        
        if response.status_code == 200:
            prop_data = response.json()
            if 'PropertyTable' in prop_data and 'Properties' in prop_data['PropertyTable']:
                props = prop_data['PropertyTable']['Properties'][0]
                info['name'] = props.get('IUPACName')
                info['formula'] = props.get('MolecularFormula')
                
                # MolecularWeight may be returned as string
                mw = props.get('MolecularWeight')
                if mw is not None:
                    try:
                        info['mw'] = float(mw)
                    except (ValueError, TypeError):
                        info['mw'] = None
        
        # Step 3: Get synonyms (common names)
        url_synonyms = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
        response = requests.get(url_synonyms, timeout=timeout)
        
        if response.status_code == 200:
            syn_data = response.json()
            if 'InformationList' in syn_data and 'Information' in syn_data['InformationList']:
                synonyms = syn_data['InformationList']['Information'][0].get('Synonym', [])
                # Filter out non-useful synonyms (too long, database codes, etc.)
                info['synonyms'] = [
                    s for s in synonyms[:15] 
                    if len(s) < 50 and not s.startswith('SCHEMB') and not s.startswith('CHEMBL')
                ]
                
                # Use first synonym as the "common name" if available
                if info['synonyms']:
                    info['name'] = info['synonyms'][0]
                    info['source'] = 'PubChem Common Name'
                elif info['name']:
                    info['source'] = 'IUPAC Name'
        
        return info
        
    except requests.exceptions.Timeout:
        return info
    except requests.exceptions.RequestException:
        return info
    except Exception:
        return info
