import type {
  DiagnoseRequest, DiagnoseResponse, ExplainResponse,
  DiseaseInfo, NLPResult, GraphData,
  BreastCancerRequest, ParkinsonsRequest, CommonDiseaseResponse,
  FeatureMetadata, ScanAnalysisResponse, MammogramAnalysisResult,
  EhrExtractionResult, AbhaBeneficiarySummary, AbhaPatientProfile,
  QuantumEscalateRequest, QuantumEscalateResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Request failed: ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  /**
   * Get the full disease catalog (common + rare)
   */
  getDiseases: async (): Promise<DiseaseInfo[]> => {
    return request<DiseaseInfo[]>(`${API_BASE_URL}/diseases`);
  },

  /**
   * Submit patient symptoms to get a diagnosis
   */
  diagnose: async (req: DiagnoseRequest): Promise<DiagnoseResponse> => {
    return request<DiagnoseResponse>(`${API_BASE_URL}/diagnose`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
  },

  /**
   * Get explainability (SHAP & next test) for a specific case
   */
  explain: async (caseId: string): Promise<ExplainResponse> => {
    return request<ExplainResponse>(`${API_BASE_URL}/explain/${caseId}`);
  },

  /**
   * Extract HPO terms from clinical free text via NLP
   */
  extractNLP: async (text: string): Promise<NLPResult[]> => {
    return request<NLPResult[]>(`${API_BASE_URL}/nlp/extract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
  },

  /**
   * Get knowledge graph data for a set of HPO terms
   */
  getGraph: async (hpoTerms: string[]): Promise<GraphData> => {
    return request<GraphData>(`${API_BASE_URL}/graph`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hpo_terms: hpoTerms }),
    });
  },

  /**
   * Get feature names and defaults for common diseases
   */
  getFeatures: async (diseaseType: string): Promise<FeatureMetadata> => {
    return request<FeatureMetadata>(`${API_BASE_URL}/diseases/${diseaseType}/features`);
  },

  /**
   * Diagnose breast cancer from 30 cellular features
   */
  diagnoseBreastCancer: async (req: BreastCancerRequest): Promise<CommonDiseaseResponse> => {
    return request<CommonDiseaseResponse>(`${API_BASE_URL}/diagnose/breast-cancer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
  },

  /**
   * Diagnose Parkinson's from 22 voice features
   */
  diagnoseParkinsons: async (req: ParkinsonsRequest): Promise<CommonDiseaseResponse> => {
    return request<CommonDiseaseResponse>(`${API_BASE_URL}/diagnose/parkinsons`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
  },

  /**
   * Get benchmark report (quantum vs classical comparison)
   */
  getBenchmarkReport: async (): Promise<any> => {
    return request<any>(`${API_BASE_URL}/benchmark/report`);
  },

  /**
   * ERR-03 / ACTION C: Live quantum escalation — replaces frontend setTimeout mockup
   */
  quantumEscalate: async (req: QuantumEscalateRequest): Promise<QuantumEscalateResponse> => {
    return request<QuantumEscalateResponse>(`${API_BASE_URL}/quantum/escalate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
  },

  /**
   * Analyze radiological medical scan (X-Ray, Echo, MRI, CT) via Computer Vision
   */
  analyzeScan: async (fileOrBase64: File | string, filename?: string): Promise<ScanAnalysisResponse> => {
    if (typeof fileOrBase64 === 'string') {
      return request<ScanAnalysisResponse>(`${API_BASE_URL}/scan/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: fileOrBase64, filename }),
      });
    }

    const formData = new FormData();
    formData.append('file', fileOrBase64);
    const response = await fetch(`${API_BASE_URL}/scan/analyze`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Scan analysis failed: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Analyze digital mammogram (extracts 30 Wisconsin features + Calibrated XGBoost + SHAP)
   */
  analyzeMammogram: async (fileOrBase64: File | string, filename?: string): Promise<MammogramAnalysisResult> => {
    if (typeof fileOrBase64 === 'string') {
      return request<MammogramAnalysisResult>(`${API_BASE_URL}/scan/mammogram`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: fileOrBase64, filename }),
      });
    }

    const formData = new FormData();
    formData.append('file', fileOrBase64);
    const response = await fetch(`${API_BASE_URL}/scan/mammogram`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Mammography analysis failed: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Extract clinical text, sections, and HPO terms from an uploaded EHR PDF
   */
  extractEhrPdf: async (fileOrBase64: File | string): Promise<EhrExtractionResult> => {
    if (typeof fileOrBase64 === 'string') {
      return request<EhrExtractionResult>(`${API_BASE_URL}/ehr/extract-pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pdf_base64: fileOrBase64 }),
      });
    }

    const formData = new FormData();
    formData.append('file', fileOrBase64);
    const response = await fetch(`${API_BASE_URL}/ehr/extract-pdf`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `EHR PDF Extraction failed: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Get direct download URL for sample clinical EHR PDF
   */
  getSampleEhrPdfUrl: (caseType = 'marfan'): string => {
    return `${API_BASE_URL}/ehr/sample-pdf?case=${caseType}`;
  },

  /**
   * Get direct download URL for a generated clinical diagnostic report PDF
   */
  getReportPdfUrl: (caseId: string): string => {
    return `${API_BASE_URL}/report/pdf/${caseId}`;
  },

  /**
   * Get direct download URL for a sample demonstration clinical diagnostic report PDF
   */
  getDemoReportPdfUrl: (): string => {
    return `${API_BASE_URL}/report/demo-pdf`;
  },

  /**
   * List verified sandbox beneficiaries from ABDM / API Setu
   */
  getAbhaBeneficiaries: async (): Promise<AbhaBeneficiarySummary[]> => {
    return request<AbhaBeneficiarySummary[]>(`${API_BASE_URL}/abha/beneficiaries`);
  },

  /**
   * Fetch patient record and FHIR R4 conditions via ABDM / ABHA Gateway
   */
  getAbhaPatient: async (abhaId: string): Promise<AbhaPatientProfile> => {
    return request<AbhaPatientProfile>(`${API_BASE_URL}/abha/patient/${encodeURIComponent(abhaId)}`);
  },
};
