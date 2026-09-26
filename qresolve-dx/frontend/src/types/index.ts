export type DiagnoseRequest = {
  symptoms: string[];
  case_id?: string;
};

export type DiagnosisResult = {
  disease: string;
  probability: number;
  rank: number;
};

export type DiagnoseResponse = {
  case_id: string;
  ranked_diagnoses: DiagnosisResult[];
  confidence: number;
  is_hard_case: boolean;
  quantum_used: boolean;
  quantum_status: string;
  top_diagnosis: string;
  runner_up: string;
};

// ERR-03 / ACTION C: Dedicated quantum escalation types
export type QuantumEscalateRequest = {
  case_id: string;
  top_diagnosis: string;
  runner_up: string;
};

export type QuantumEscalateResponse = {
  case_id: string;
  quantum_status: string;
  ranked_diagnoses: DiagnosisResult[];
  confidence: number;
  top_diagnosis: string;
  runner_up: string;
  circuit_metrics?: {
    n_qubits: number;
    kernel_type: string;
    entanglement: string;
    reps: number;
    computation_time_s: number;
    pre_trained: boolean;
  };
};

export type EvidenceItem = {
  hpo_id: string;
  label: string;
  direction: string;
  shap_value: number;
};

export type NextTest = {
  hpo_id: string;
  label: string;
  clinical_test: string;
  information_gain: number;
  expected_outcome_positive: string;
  expected_outcome_negative: string;
};

export type ExplainResponse = {
  case_id: string;
  top_diagnosis: string;
  runner_up: string;
  supporting_evidence: EvidenceItem[];
  against_evidence: EvidenceItem[];
  suggested_tests: NextTest[];
};

export type PatientCase = {
  id: string;
  patientName: string;
  age: number;
  sex: string;
  currentStage: number; // 1-9
  diagnoseResponse?: DiagnoseResponse;
  explainResponse?: ExplainResponse;
};

export type BreastCancerRequest = {
  features: Record<string, number>;
};

export type ParkinsonsRequest = {
  features: Record<string, number>;
};

export type CommonDiseaseResponse = {
  diagnosis: string;
  probability: number;
  confidence: string;
  supporting_evidence?: any[];
  against_evidence?: any[];
};

export type DiseaseInfo = {
  id: string;
  name: string;
  category: 'common' | 'rare';
  track: 'common' | 'classical' | 'quantum';
  description: string;
  genes: string[];
  n_symptoms: number;
};

export type NLPResult = {
  hpo_id: string;
  label: string;
  confirmed: boolean;
};

export type GraphNode = {
  id: string;
  group: 'disease' | 'symptom';
  label: string;
};

export type GraphLink = {
  source: string;
  target: string;
  type?: string;
};

export type GraphData = {
  nodes: GraphNode[];
  links: GraphLink[];
};

export type FeatureMetadata = {
  feature_names: string[];
  defaults: Record<string, number>;
};

export type ScanFindingItem = {
  hpo_id: string;
  label: string;
  confidence: number;
  modality: string;
  anatomical_region: string;
  clinical_evidence: string;
};

export type MammogramAnalysisResult = {
  modality: string;
  diagnosis: string;
  probability: number;
  confidence: string;
  birads_score: string;
  extracted_features: Record<string, number>;
  lesion_metrics: Record<string, number>;
  supporting_evidence: { feature: string; value: number; shap_value: number }[];
  against_evidence: { feature: string; value: number; shap_value: number }[];
  radiological_summary: string;
};

export type ScanAnalysisResponse = {
  detected_hpo_ids: string[];
  findings: ScanFindingItem[];
  modality_detected: string;
  radiomic_metrics: Record<string, number>;
  breast_cancer_diagnosis?: MammogramAnalysisResult;
  extracted_features?: Record<string, number>;
};

export type EhrExtractionResult = {
  full_text: string;
  page_count: number;
  sections: Record<string, string>;
  metadata: Record<string, string>;
  extracted_hpo_terms: Array<{ hpo_id: string; label: string; confirmed: boolean }>;
  recommended_route: 'classical' | 'quantum';
};

export type AbhaBeneficiarySummary = {
  abha_id: string;
  name: string;
  gender: string;
  age: number;
  beneficiary_scheme: string;
  facility_name: string;
  clinical_summary: string;
  n_conditions: number;
};

export type AbhaPatientProfile = {
  abha_id: string;
  abha_address: string;
  name: string;
  gender: string;
  age: number;
  beneficiary_scheme: string;
  facility_name: string;
  last_updated: string;
  clinical_summary: string;
  fhir_conditions: Array<{ code: string; display: string }>;
  hpo_ids: string[];
  fhir_bundle: any;
};
