from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Fragments
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from rdkit.Chem import AllChem

def swap_methyl_for_alcohol(smiles):
    """
    Replace methyl groups (CH3) with amine groups (NH2) using reaction SMARTS.
    
    Args:
        smiles: SMILES string of the input molecule
    
    Returns:
        str: SMILES string with methyl groups replaced by amines, or None if conversion fails
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    
    try:
        # Reaction SMARTS: replace methyl carbon with nitrogen
        # [C:1]([H])([H])([H]) matches a carbon with 3 hydrogens
        # The reaction replaces it with [N:1]([H])([H]) - nitrogen with 2 hydrogens
        reaction_smarts = '[C:1]([H])([H])([H]) >> [O:1]([H])'
        reaction = AllChem.ReactionFromSmarts(reaction_smarts)
        
        if reaction is None:
            return None
        
        # Apply the reaction
        products = reaction.RunReactants((mol,))
        
        if not products:
            # No methyl groups found, return original SMILES
            return smiles
        
        # Get the first product
        new_mol = products[0][0]
        
        # Sanitize the molecule
        Chem.SanitizeMol(new_mol)
        
        # Convert back to SMILES
        new_mol = Chem.RemoveHs(new_mol)
        new_smiles = Chem.MolToSmiles(new_mol)

        return new_smiles
        
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None


def swap_ethyl_for_hydroxyl_amine(smiles):
    """
    Replace ethyl groups (CH2CH3) with hydroxyl amine groups (NOH) using reaction SMARTS.
    
    Args:
        smiles: SMILES string of the input molecule
    
    Returns:
        str: SMILES string with ethyl groups replaced by hydroxyl amines, or None if conversion fails
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    
    try:
        # Reaction SMARTS: replace ethyl group with hydroxyl amine
        reaction_smarts = '[C:1]([H])([H])[C:2]([H])([H])([H]) >> [N:1][O:2][H]'
        reaction = AllChem.ReactionFromSmarts(reaction_smarts)
        
        if reaction is None:
            return None
        
        # Apply the reaction
        products = reaction.RunReactants((mol,))
        
        if not products:
            # No ethyl groups found, return original SMILES
            return smiles
        
        # Get the first product
        new_mol = products[0][0]
        
        # Sanitize the molecule
        try:
            Chem.SanitizeMol(new_mol)
        except ValueError:
            # If sanitization fails, try adjusting the hydrogen counts
            new_mol = Chem.RenumberAtoms(new_mol)
            Chem.SanitizeMol(new_mol)
        
        # Convert back to SMILES
        new_mol = Chem.RemoveHs(new_mol)
        new_smiles = Chem.MolToSmiles(new_mol)

        return new_smiles
        
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None


def swap_ketone_for_carboxylic_acid(smiles):
    """
    Replace ketone groups (C=O) with carboxylic acid groups (C(=O)O) using reaction SMARTS.
    
    Args:
        smiles: SMILES string of the input molecule
    
    Returns:
        str: SMILES string with ketone groups replaced by carboxylic acids, or None if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        
        # Define reaction SMARTS for ketone to carboxylic acid conversion
        # This pattern matches ketones (C=O where C is not connected to another O or N)
        ketone_pattern = "[C:1](=[O:2])[C,H:3]"
        carboxylic_product = "[C:1](=[O:2])O"
        
        # Create the reaction
        reaction_smarts = f"{ketone_pattern}>>{carboxylic_product}"
        rxn = AllChem.ReactionFromSmarts(reaction_smarts)
        
        if rxn is None:
            print(f"Failed to create reaction from SMARTS: {reaction_smarts}")
            return None
        
        # Apply the reaction
        products = rxn.RunReactants((mol,))
        
        if not products:
            # No ketone groups found to convert
            return smiles
        
        # Get the first product (there should only be one)
        product_mol = products[0][0]
        
        # Remove explicit hydrogens and sanitize
        product_mol = Chem.RemoveHs(product_mol)
        Chem.SanitizeMol(product_mol)
        
        # Convert back to SMILES
        product_smiles = Chem.MolToSmiles(product_mol)
        
        return product_smiles
        
    except Exception as e:
        print(f"Error in swap_ketone_for_carboxylic_acid for {smiles}: {e}")
        return None


def swap_methyl_for_amine(smiles):
    """
    Replace methyl groups (CH3) with amine groups (NH2) using reaction SMARTS.
    
    Args:
        smiles: SMILES string of the input molecule
    
    Returns:
        str: SMILES string with methyl groups replaced by amines, or None if conversion fails
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    
    try:
        # Reaction SMARTS: replace methyl carbon with nitrogen
        # [C:1]([H])([H])([H]) matches a carbon with 3 hydrogens
        # The reaction replaces it with [N:1]([H])([H]) - nitrogen with 2 hydrogens
        reaction_smarts = '[C:1]([H])([H])([H]) >> [N:1]([H])([H])'
        reaction = AllChem.ReactionFromSmarts(reaction_smarts)
        
        if reaction is None:
            return None
        
        # Apply the reaction
        products = reaction.RunReactants((mol,))
        
        if not products:
            # No methyl groups found, return original SMILES
            return smiles
        
        # Get the first product
        new_mol = products[0][0]
        
        # Sanitize the molecule
        Chem.SanitizeMol(new_mol)
        
        # Convert back to SMILES
        new_mol = Chem.RemoveHs(new_mol)
        new_smiles = Chem.MolToSmiles(new_mol)

        return new_smiles
        
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None


def swap_ethyl_for_hydroxyl_amide(smiles):
    """
    Replace ethyl groups (CH2CH3) with hydroxyl amide groups (OHNH) using reaction SMARTS.
    
    Args:
        smiles: SMILES string of the input molecule
    
    Returns:
        str: SMILES string with ethyl groups replaced by hydroxyl amides, or None if conversion fails
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    
    try:
        # Reaction SMARTS: replace ethyl group with hydroxyl amide
        reaction_smarts = '[C:1]([H])([H])[C:2]([H])([H])([H]) >> [O:1][N:2][H]'
        reaction = AllChem.ReactionFromSmarts(reaction_smarts)
        
        if reaction is None:
            return None
        
        # Apply the reaction
        products = reaction.RunReactants((mol,))
        
        if not products:
            # No ethyl groups found, return original SMILES
            return smiles
        
        # Get the first product
        new_mol = products[0][0]
        
        # Sanitize the molecule
        try:
            Chem.SanitizeMol(new_mol)
        except ValueError:
            # If sanitization fails, try adjusting the hydrogen counts
            new_mol = Chem.RenumberAtoms(new_mol)
            Chem.SanitizeMol(new_mol)
        
        # Convert back to SMILES
        new_mol = Chem.RemoveHs(new_mol)
        new_smiles = Chem.MolToSmiles(new_mol)

        return new_smiles
        
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None

def swap_ketone_for_imine(smiles):
    """
    Replace ketone groups (C=O) with imine groups (C=N) using reaction SMARTS.
    
    Args:
        smiles: SMILES string of the input molecule
    
    Returns:
        str: SMILES string with ketone groups replaced by imines, or None if conversion fails
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    
    try:
        # Reaction SMARTS: replace ketone group with imine
        reaction_smarts = '[C:1](=O) >> [C:1](=[N])'
        reaction = AllChem.ReactionFromSmarts(reaction_smarts)
        
        if reaction is None:
            return None
        
        # Apply the reaction
        products = reaction.RunReactants((mol,))
        
        if not products:
            # No ketone groups found, return original SMILES
            return smiles
        
        # Get the first product
        new_mol = products[0][0]
        
        # Sanitize the molecule
        try:
            Chem.SanitizeMol(new_mol)
        except ValueError:
            # If sanitization fails, try adjusting the hydrogen counts
            new_mol = Chem.RenumberAtoms(new_mol)
            Chem.SanitizeMol(new_mol)
        
        # Convert back to SMILES
        new_mol = Chem.RemoveHs(new_mol)
        new_smiles = Chem.MolToSmiles(new_mol)

        return new_smiles
        
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None


def apply_molecular_transformations(input_csv_path, output_csv_path):
    """
    Apply molecular transformations to SMILES in a CSV file and save only the transformed results.
    
    Transformations applied:
    1. Swap methyl groups for amine groups
    2. Swap ethyl groups for hydroxyl amide groups  
    3. Swap ketone groups for imine groups
    
    Args:
        input_csv_path: Path to input CSV file with 'SMILES' column
        output_csv_path: Path to save the transformed results
    
    Returns:
        pd.DataFrame: DataFrame containing only transformed SMILES
    """
    # Read the input CSV
    df = pd.read_csv(input_csv_path)
    
    if 'SMILES' not in df.columns:
        raise ValueError("CSV must contain a column named 'SMILES'")
    
    print(f"Processing {len(df)} molecules...")
    
    # Initialize list to store transformed SMILES
    transformed_smiles = []
    
    for idx, row in df.iterrows():
        original_smiles = row['SMILES']
        
        if pd.isna(original_smiles):
            continue
            
        # Apply transformations sequentially
        current_smiles = original_smiles
        was_transformed = False
        
        # # Transformation 1: Methyl to Amine
        # methyl_to_amine = swap_methyl_for_amine(current_smiles)
        # if methyl_to_amine and methyl_to_amine != current_smiles:
        #     current_smiles = methyl_to_amine
        #     was_transformed = True

        # Transformation 1: Methyl to Alcohol
        methyl_to_alcohol = swap_methyl_for_alcohol(current_smiles)
        if methyl_to_alcohol and methyl_to_alcohol != current_smiles:
            current_smiles = methyl_to_alcohol
            was_transformed = True

        
        # # Transformation 2: Ethyl to Hydroxyl Amide
        # ethyl_to_hydroxyl = swap_ethyl_for_hydroxyl_amide(current_smiles)
        # if ethyl_to_hydroxyl and ethyl_to_hydroxyl != current_smiles:
        #     current_smiles = ethyl_to_hydroxyl
        #     was_transformed = True

        # Transformation 2: Ethyl to Hydroxyl Amine (R–N–OH)
        ethyl_to_hydroxyl_amine = swap_ethyl_for_hydroxyl_amine(current_smiles)
        if ethyl_to_hydroxyl_amine and ethyl_to_hydroxyl_amine != current_smiles:
            current_smiles = ethyl_to_hydroxyl_amine
            was_transformed = True
        
        # # Transformation 3: Ketone to Imine
        # ketone_to_imine = swap_ketone_for_imine(current_smiles)
        # if ketone_to_imine and ketone_to_imine != current_smiles:
        #     current_smiles = ketone_to_imine
        #     was_transformed = True

        # Transformation 3: Ketone to Carboxylic Acid (add OH to the carbonyl carbon)
        ketone_to_carboxylic = swap_ketone_for_carboxylic_acid(current_smiles)
        if ketone_to_carboxylic and ketone_to_carboxylic != current_smiles:
            current_smiles = ketone_to_carboxylic
            was_transformed = True

        
        # Only add if the molecule was actually transformed
        if was_transformed:
            transformed_smiles.append(current_smiles)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} molecules...")
    
    # Remove duplicates
    transformed_smiles = list(set(transformed_smiles))
    
    # Create DataFrame with only transformed SMILES
    result_df = pd.DataFrame(transformed_smiles, columns=['SMILES'])
    
    # Save to CSV
    result_df.to_csv(output_csv_path, index=False)
    
    # Print summary
    print(f"\nTransformation Summary:")
    print(f"Total molecules processed: {len(df)}")
    print(f"Unique transformed molecules: {len(result_df)}")
    print(f"Results saved to: {output_csv_path}")
    
    return result_df


def swap_methyl_for_thiol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        reaction = AllChem.ReactionFromSmarts('[C:1]([H])([H])([H]) >> [S:1][H]')
        if reaction is None:
            return None
        products = reaction.RunReactants((mol,))
        if not products:
            return smiles
        new_mol = products[0][0]
        Chem.SanitizeMol(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        return Chem.MolToSmiles(new_mol)
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None


def swap_halide_for_sulfonyl(smiles):
    """Replace a halide (F, Cl, Br) with a sulfonyl group (SO2H)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        reaction = AllChem.ReactionFromSmarts('[F,Cl,Br:1] >> [S:1](=O)(=O)O')
        if reaction is None:
            return None
        products = reaction.RunReactants((mol,))
        if not products:
            return smiles
        new_mol = products[0][0]
        Chem.SanitizeMol(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        return Chem.MolToSmiles(new_mol)
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None


def swap_halide_for_disulfide(smiles):
    """Replace a halide (F, Cl, Br) with a persulfide group (S-SH)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        reaction = AllChem.ReactionFromSmarts('[F,Cl,Br:1] >> [S:1]S')
        if reaction is None:
            return None
        products = reaction.RunReactants((mol,))
        if not products:
            return smiles
        new_mol = products[0][0]
        Chem.SanitizeMol(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        return Chem.MolToSmiles(new_mol)
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None


def swap_amine_for_sulfinyl(smiles):
    """Replace a primary amine (NH2) with a sulfinyl group (S(=O)H)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        reaction = AllChem.ReactionFromSmarts('[N:1]([H])([H]) >> [S:1]([H])(=O)')
        if reaction is None:
            return None
        products = reaction.RunReactants((mol,))
        if not products:
            return smiles
        new_mol = products[0][0]
        Chem.SanitizeMol(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        return Chem.MolToSmiles(new_mol)
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None


def apply_sulfur_augmentation(input_csv_path, output_csv_path):
    """
    Apply SH-generating transformations to SMILES in a CSV file.

    Each swap is applied independently to the original molecule — a molecule
    with multiple qualifying functional groups generates one augmented variant
    per applicable swap.

    Transformations applied:
    1. Methyl → thiol (SH)
    2. Halide → sulfonyl (SO2H)
    3. Halide → persulfide (S-SH)
    4. Primary amine → sulfinyl (S(=O)H)
    """
    df = pd.read_csv(input_csv_path)

    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    if smiles_col not in df.columns:
        raise ValueError("CSV must contain a 'smiles' or 'SMILES' column")

    swaps = [
        swap_methyl_for_thiol,
        swap_halide_for_sulfonyl,
        swap_halide_for_disulfide,
        swap_amine_for_sulfinyl,
    ]

    print(f"Processing {len(df)} molecules with {len(swaps)} swap functions...")

    augmented = []
    for idx, row in df.iterrows():
        original = row[smiles_col]
        if pd.isna(original):
            continue
        for fn in swaps:
            result = fn(original)
            if result and result != original:
                augmented.append(result)
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} molecules...")

    augmented = list(set(augmented))
    result_df = pd.DataFrame(augmented, columns=['smiles'])
    result_df.to_csv(output_csv_path, index=False)

    print(f"\nAugmentation Summary:")
    print(f"Input molecules:           {len(df)}")
    print(f"Unique augmented molecules: {len(result_df)}")
    print(f"Results saved to:          {output_csv_path}")

    return result_df
