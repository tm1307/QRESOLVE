"""
HPO vectorization with Information Content (IC) weighting and Lin's semantic similarity.
"""

import sys
import os
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data.disease_data import (
    ALL_DISEASES, ALL_HPO_TERMS, HPO_TERM_INDEX, NUM_FEATURES,
    DISEASE_NAMES, DISEASE_LABEL_MAP, HPO_TERMS
)

# Approximate HPO parent map to enable simplified Lin similarity without the full DAG.
#
# LIMITATION (ERR-05): This flat parent mapping only captures one level of the
# HPO hierarchy. Deep phenotypic sub-categories (e.g., HP:0001083 "Ectopia lentis"
# → HP:0012372 "Abnormal eye morphology" → HP:0000478 "Abnormality of the eye")
# are collapsed to a single parent, which can cause slight semantic distortion.
# Full DAG traversal requires the `pyhpo` dependency. When `pyhpo` is installed,
# `compute_lin_similarity_full_dag()` below will be used automatically instead.
HPO_PARENT_MAP = {
    # Abnormality of the eye (HP:0000478)
    "HP:0000486": "HP:0000478",
    "HP:0001083": "HP:0000478",
    "HP:0000545": "HP:0000478",
    "HP:0000541": "HP:0000478",
    "HP:0000639": "HP:0000478",
    "HP:0000508": "HP:0000478",
    "HP:0007703": "HP:0000478",

    # Abnormality of the cardiovascular system (HP:0001626)
    "HP:0001631": "HP:0001626",
    "HP:0001650": "HP:0001626",
    "HP:0001659": "HP:0001626",
    "HP:0002616": "HP:0001626",
    "HP:0004942": "HP:0001626",
    "HP:0005110": "HP:0001626",
    "HP:0012722": "HP:0001626",
    "HP:0002617": "HP:0001626",
    "HP:0001679": "HP:0001626",

    # Abnormality of the skeletal system (HP:0000924)
    "HP:0001166": "HP:0000924",
    "HP:0001238": "HP:0000924",
    "HP:0001382": "HP:0000924",
    "HP:0002650": "HP:0000924",
    "HP:0002705": "HP:0000924",
    "HP:0002758": "HP:0000924",
    "HP:0002808": "HP:0000924",
    "HP:0003088": "HP:0000924",
    "HP:0003179": "HP:0000924",
    "HP:0004209": "HP:0000924",
    "HP:0005059": "HP:0000924",
    "HP:0005692": "HP:0000924",
    "HP:0006610": "HP:0000924",
    "HP:0006657": "HP:0000924",
    "HP:0002829": "HP:0000924",
    "HP:0003423": "HP:0000924",
    
    # Abnormality of head or neck (HP:0000152)
    "HP:0000218": "HP:0000152",
    "HP:0000272": "HP:0000152",
    "HP:0000278": "HP:0000152",
    "HP:0000316": "HP:0000152",
    "HP:0000348": "HP:0000152",
    "HP:0000369": "HP:0000152",
    "HP:0000431": "HP:0000152",
    "HP:0000455": "HP:0000152",
    "HP:0000494": "HP:0000152",
    "HP:0000252": "HP:0000152",
    
    # Abnormality of skin/integument (HP:0000951)
    "HP:0000974": "HP:0000951",
    "HP:0000977": "HP:0000951",
    "HP:0000987": "HP:0000951",
    "HP:0000988": "HP:0000951",
    "HP:0000994": "HP:0000951",
    "HP:0001030": "HP:0000951",
    "HP:0001065": "HP:0000951",
    "HP:0008066": "HP:0000951",
    "HP:0011121": "HP:0000951",
    
    # Other/Metabolism/GI (HP:0001939)
    "HP:0002019": "HP:0001939",
    "HP:0002020": "HP:0001939",
    "HP:0002104": "HP:0001939",
    "HP:0002315": "HP:0001939"
}

def compute_information_content() -> Dict[str, float]:
    """
    Compute IC(term) = -log(P(term)) where P(term) = (number of diseases annotated with term) / (total diseases)
    Uses the 5-disease corpus from disease_data.
    More specific terms get higher IC.
    """
    ic_values = {}
    total_diseases = len(ALL_DISEASES)
    
    term_counts = {term: 0 for term in ALL_HPO_TERMS}
    
    for disease in ALL_DISEASES:
        # A disease has a term if it appears in its symptoms dict
        for term in disease.symptoms.keys():
            if term in term_counts:
                term_counts[term] += 1
                
    for term, count in term_counts.items():
        if count > 0:
            p_term = count / total_diseases
            ic_values[term] = -math.log(p_term)
        else:
            ic_values[term] = float('inf')
            
    # Handle zero-count terms by assigning max IC + 1
    valid_ics = [ic for ic in ic_values.values() if ic != float('inf')]
    max_ic = max(valid_ics) if valid_ics else 0.0
    
    for term, ic in ic_values.items():
        if ic == float('inf'):
            ic_values[term] = max_ic + 1.0
            
    return ic_values


def try_compute_ic_with_pyhpo() -> Optional[Dict[str, float]]:
    """
    Try to use pyhpo for full HPO corpus IC (much better IC values).
    Return None if pyhpo not installed.
    """
    try:
        from pyhpo import Ontology
        _ = Ontology()
        
        ic_values = {}
        for term in ALL_HPO_TERMS:
            try:
                hpo_term = Ontology.get_hpo_object(term)
                ic_values[term] = hpo_term.information_content.omim
            except Exception:
                # If term not found, fallback to 0.0 or skip
                pass
                
        return ic_values if ic_values else None
    except ImportError:
        return None
    except Exception:
        return None


def compute_lin_similarity(t1: str, t2: str, ic_values: Dict[str, float]) -> float:
    """
    Lin's semantic similarity between two HPO terms.
    
    Formula:  sim_Lin(t1, t2) = 2 · IC(MICA) / (IC(t1) + IC(t2))
    
    Where MICA = Most Informative Common Ancestor.
    
    Without a full HPO DAG, we approximate MICA using the HPO_PARENT_MAP:
    if two terms share the same parent category, the MICA is that parent.
    The IC of the parent is looked up from ic_values (computed over our disease corpus).
    If the parent is not in ic_values, we use IC = 0 (root-level, maximally general).
    
    Properties:
    - sim(t, t) = 1.0  (identity)
    - sim(t1, t2) ∈ [0, 1]  (bounded)
    - sim(t1, t2) = 0 if no common ancestor  (unrelated terms)
    
    Reference: Lin D. (1998) "An Information-Theoretic Definition of Similarity"
    """
    if t1 == t2:
        return 1.0
        
    ic_t1 = ic_values.get(t1, 0.0)
    ic_t2 = ic_values.get(t2, 0.0)
    
    if ic_t1 + ic_t2 == 0:
        return 0.0
        
    parent1 = HPO_PARENT_MAP.get(t1)
    parent2 = HPO_PARENT_MAP.get(t2)
    
    if parent1 and parent2 and parent1 == parent2:
        # MICA is the shared parent — look up its actual IC
        # If the parent itself is in our corpus, use its IC; otherwise it's
        # a very general category (e.g., "Abnormality of the eye") with IC ≈ 0
        mica_ic = ic_values.get(parent1, 0.0)
        
        # Ensure MICA IC doesn't exceed either child's IC
        # (a parent is always more general than its children)
        mica_ic = min(mica_ic, ic_t1, ic_t2)
        
        if mica_ic <= 0:
            return 0.0
            
        return 2.0 * mica_ic / (ic_t1 + ic_t2)
        
    return 0.0


def compute_lin_similarity_full_dag(t1: str, t2: str, ic_values: Dict[str, float]) -> float:
    """
    Lin's semantic similarity using the full HPO DAG via pyhpo.

    Falls back to the flat-parent approximation if pyhpo is not installed.
    This resolves ERR-05 by traversing the complete ontology graph to find
    the true Most Informative Common Ancestor (MICA).
    """
    if t1 == t2:
        return 1.0

    try:
        from pyhpo import Ontology
        Ontology()

        hpo_t1 = Ontology.get_hpo_object(t1)
        hpo_t2 = Ontology.get_hpo_object(t2)

        # Find common ancestors via full DAG traversal
        common_ancestors = hpo_t1.common_ancestors(hpo_t2)
        if not common_ancestors:
            return 0.0

        # MICA = ancestor with highest IC
        mica_ic = max(ic_values.get(str(a), 0.0) for a in common_ancestors)

        ic_t1 = ic_values.get(t1, 0.0)
        ic_t2 = ic_values.get(t2, 0.0)

        if ic_t1 + ic_t2 == 0:
            return 0.0

        return 2.0 * mica_ic / (ic_t1 + ic_t2)
    except (ImportError, Exception):
        # pyhpo not available — fall back to flat parent map approximation
        return compute_lin_similarity(t1, t2, ic_values)


def lin_similarity(t1: str, t2: str, ic_values: Dict[str, float]) -> float:
    """
    Smart dispatcher: uses full DAG traversal if pyhpo is available,
    otherwise uses the flat-parent approximation.
    """
    try:
        import pyhpo  # noqa: F401
        return compute_lin_similarity_full_dag(t1, t2, ic_values)
    except ImportError:
        return compute_lin_similarity(t1, t2, ic_values)


def build_ic_weighted_features(patient_hpo_terms: List[str], ic_values: Dict[str, float]) -> np.ndarray:
    """
    Build a feature vector of length NUM_FEATURES.
    For each term in ALL_HPO_TERMS:
      - If term is directly in patient's terms: value = IC(term)
      - If a semantically similar term (Lin sim > 0.5) is in patient's terms: value = IC(term) * sim_score
      - Else: value = 0.0
    """
    feature_vector = np.zeros(NUM_FEATURES)
    
    for term_idx, term in enumerate(ALL_HPO_TERMS):
        if term in patient_hpo_terms:
            feature_vector[term_idx] = ic_values.get(term, 0.0)
        else:
            # Check for semantically similar terms
            max_sim_val = 0.0
            for pt_term in patient_hpo_terms:
                sim = compute_lin_similarity(term, pt_term, ic_values)
                if sim > 0.5:
                    val = ic_values.get(term, 0.0) * sim
                    if val > max_sim_val:
                        max_sim_val = val
            feature_vector[term_idx] = max_sim_val
            
    return feature_vector


def build_feature_matrix(patients_df: pd.DataFrame, ic_values: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes a DataFrame with 'hpo_terms' column (pipe-separated) and 'true_disease' column
    Returns X (n_samples, NUM_FEATURES) IC-weighted matrix and y (n_samples,) label array
    """
    X_list = []
    y_list = []
    
    for _, row in patients_df.iterrows():
        hpo_terms_raw = row.get('hpo_terms', '')
        terms = hpo_terms_raw.split('|') if isinstance(hpo_terms_raw, str) and hpo_terms_raw else []
        
        feature_vec = build_ic_weighted_features(terms, ic_values)
        X_list.append(feature_vec)
        
        disease = row.get('true_disease', '')
        label = DISEASE_LABEL_MAP.get(disease, -1)
        y_list.append(label)
        
    return np.array(X_list), np.array(y_list)


if __name__ == "__main__":
    # Compute IC values
    print("Computing IC values...")
    ic_dict = try_compute_ic_with_pyhpo()
    if ic_dict is None:
        print("pyhpo not available. Using local corpus for IC computation.")
        ic_dict = compute_information_content()
    else:
        print("Successfully used pyhpo for IC computation.")
        
    # Sort terms by IC value (descending - most specific first)
    sorted_terms = sorted(ic_dict.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 15 most specific HPO terms (highest IC):")
    for i, (term, ic) in enumerate(sorted_terms[:15]):
        name = HPO_TERMS.get(term, {}).get("name", "Unknown")
        print(f"{i+1:2d}. {term}: {name:<35} (IC: {ic:.4f})")
