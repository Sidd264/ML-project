"""
Machine Learning Prediction Model for Acute Oral Toxicity
Streamlit Application
"""

import streamlit as st
import pandas as pd
from utils import (
    load_model_artifacts,
    validate_smiles,
    generate_descriptors,
    prepare_features,
    predict_toxicity,
    get_compound_info
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Acute Oral Toxicity Predictor",
    page_icon="🧪",
    layout="wide"
)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function."""
    
    # Load model artifacts (cached)
    try:
        model, label_encoder, feature_columns, metadata = load_model_artifacts()
    except Exception as e:
        st.error("❌ Failed to load model. Please ensure all model files are present in the `model/` directory.")
        st.stop()
    
    # ========================================================================
    # SIDEBAR - MODEL INFORMATION
    # ========================================================================
    
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown(f"**Model:** {metadata.get('model', 'N/A')}")
        st.markdown(f"**Version:** {metadata.get('version', 'N/A')}")
        st.markdown(f"**Dataset:** {metadata.get('dataset', 'N/A')}")
        st.markdown(f"**Features:** {metadata.get('feature_count', 'N/A')}")
        
        st.markdown("---")
        st.markdown("**Performance Metrics:**")
        eval_metrics = metadata.get('evaluation', {})
        st.markdown(f"- Macro F1: **{eval_metrics.get('macro_f1', 'N/A'):.4f}**")
        st.markdown(f"- ROC-AUC: **{eval_metrics.get('roc_auc_ovr', 'N/A'):.4f}**")
        st.markdown(f"- Accuracy: **{eval_metrics.get('accuracy', 'N/A'):.4f}**")
        
        st.markdown("---")
        st.markdown("**Target Classes:**")
        for i, cls in enumerate(metadata.get('target_classes', [])):
            st.markdown(f"- Class {i}: {cls}")
        
        st.markdown("---")
        st.markdown("**About:**")
        st.markdown(
            "This application predicts acute oral toxicity of chemical compounds "
            "using molecular descriptors and a trained XGBoost classifier. "
            "Simply enter a SMILES string to get started."
        )
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    st.title("🧪 Machine Learning Prediction Model for Acute Oral Toxicity")
    
    st.markdown(
        "Predict the acute oral toxicity of chemical compounds using molecular descriptors "
        "and machine learning. Enter a valid **SMILES string** below to get started."
    )
    
    # ========================================================================
    # INPUT SECTION
    # ========================================================================
    
    st.markdown("### 📝 Input")
    
    smiles_input = st.text_input(
        "Enter SMILES string:",
        placeholder="e.g., CCN(CC)C(=O)Nc1c(C)cc(C)cc1C",
        help="Enter a valid SMILES string representing a chemical compound"
    )
    
    # Predict button
    predict_button = st.button("🔬 Predict Toxicity", type="primary", use_container_width=True)
    
    # ========================================================================
    # RESULTS SECTION
    # ========================================================================
    
    if predict_button:
        # Validate input
        if not smiles_input or not smiles_input.strip():
            st.error("⚠️ Please enter a SMILES string.")
            return
        
        # Validate SMILES
        is_valid, mol = validate_smiles(smiles_input)
        
        if not is_valid:
            st.error("❌ Invalid SMILES string. Please check your input and try again.")
            return
        
        # Generate descriptors
        try:
            with st.spinner("Generating molecular descriptors..."):
                descriptors = generate_descriptors(mol)
        except Exception as e:
            st.error(f"❌ Error generating descriptors: {str(e)}")
            return
        
        # Prepare features
        try:
            features_df = prepare_features(descriptors, feature_columns)
        except Exception as e:
            st.error(f"❌ Error preparing features: {str(e)}")
            return
        
        # Make prediction
        try:
            with st.spinner("Predicting toxicity..."):
                predicted_class, confidence, probabilities = predict_toxicity(
                    model, features_df, label_encoder
                )
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            return
        
        # ====================================================================
        # DISPLAY RESULTS
        # ====================================================================
        
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        # Fetch compound name from PubChem
        compound_info = None
        with st.spinner("Looking up compound information..."):
            compound_info = get_compound_info(smiles_input)
        
        # Create two columns for prediction card and molecule image
        col_pred, col_mol = st.columns([2, 1])
        
        with col_pred:
            # Compound name header
            if compound_info and compound_info.get('name'):
                st.markdown(f"### 🧪 {compound_info['name']}")
                
                # Show additional info if available
                info_parts = []
                if compound_info.get('formula'):
                    info_parts.append(f"**Formula:** {compound_info['formula']}")
                if compound_info.get('mw'):
                    info_parts.append(f"**MW:** {compound_info['mw']:.2f} g/mol")
                if compound_info.get('cid'):
                    info_parts.append(
                        f"[PubChem CID: {compound_info['cid']}]"
                        f"(https://pubchem.ncbi.nlm.nih.gov/compound/{compound_info['cid']})"
                    )
                
                if info_parts:
                    st.markdown(" | ".join(info_parts))
            else:
                st.markdown("### 🧪 Unknown Compound")
                st.caption("Compound name not found in PubChem database")
            
            st.markdown("---")
            
            # Prediction card
            st.markdown(f"#### **Predicted Toxicity Class:** {predicted_class}")
            st.markdown(f"**Confidence:** {confidence:.2%}")
            
            st.markdown("---")
            
            # Probability distribution
            st.markdown("**Probability Distribution:**")
            
            target_classes = metadata.get('target_classes', [
                'Highly Toxic (≤ 50)',
                'Moderately Toxic (50-2000)',
                'Low Toxic (> 2000)'
            ])
            
            prob_df = pd.DataFrame({
                'Class': target_classes,
                'Probability': probabilities
            })
            
            for idx, row in prob_df.iterrows():
                st.markdown(f"**{row['Class']}**")
                st.progress(float(row['Probability']))
                st.markdown(f"{row['Probability']:.2%}")
                st.markdown("")
        
        with col_mol:
            # Molecule structure using PubChem API
            st.markdown("#### **Molecule Structure**")
            
            if compound_info and compound_info.get('cid'):
                pubchem_image_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound_info['cid']}/PNG?image_size=300x300"
                caption_text = compound_info.get('name', smiles_input[:50])
                st.image(pubchem_image_url, caption=caption_text, use_container_width=True)
            else:
                st.warning("⚠️ Compound not found in PubChem database. Molecule visualization is unavailable, but the prediction remains valid.")
        
        # ====================================================================
        # DETAILED RESULTS EXPANDER
        # ====================================================================
        
        with st.expander("📊 View Detailed Results"):
            st.markdown("**Input SMILES:**")
            st.code(smiles_input)
            
            if compound_info:
                st.markdown("**Compound Information (from PubChem):**")
                if compound_info.get('synonyms'):
                    st.markdown(f"**Common Names:** {', '.join(compound_info['synonyms'][:5])}")
            
            st.markdown("**Generated Descriptors (First 10):**")
            desc_df = pd.DataFrame(
                list(descriptors.items())[:10], 
                columns=['Descriptor', 'Value']
            )
            st.dataframe(desc_df, use_container_width=True)
            
            st.markdown(f"**Total Descriptors Generated:** {len(descriptors)}")
            st.markdown(f"**Features Used by Model:** {len(feature_columns)}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
