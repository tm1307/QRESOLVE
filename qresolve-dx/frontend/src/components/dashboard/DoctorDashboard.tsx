import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../../lib/api';
import type { DiseaseInfo, ScanAnalysisResponse } from '../../types';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';

export const DoctorDashboard = () => {
  const navigate = useNavigate();
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Scan analysis state for dashboard upload
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState<ScanAnalysisResponse | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDiseases = async () => {
      try {
        const data = await api.getDiseases();
        setDiseases(data);
      } catch (err: any) {
        setError(err.message || 'Failed to fetch diseases');
      } finally {
        setLoading(false);
      }
    };
    fetchDiseases();
  }, []);

  const handleScanUpload = async (file: File) => {
    setIsScanning(true);
    setScanError(null);
    try {
      const result = await api.analyzeScan(file, file.name);
      setScanResult(result);
    } catch (err: any) {
      setScanError(err.message || 'Failed to analyze imaging scan');
    } finally {
      setIsScanning(false);
    }
  };

  const handlePresetScan = async (presetType: 'chest_xray' | 'echo' | 'mra' | 'mammogram') => {
    setIsScanning(true);
    setScanError(null);
    try {
      const dummyPngBase64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
      const filenameMap = {
        chest_xray: "chest_radiograph_pectus.png",
        echo: "echocardiogram_aortic_dilation.png",
        mra: "mra_angiography_tortuosity.png",
        mammogram: "mammogram_breast_calcification.png",
      };
      const result = await api.analyzeScan(dummyPngBase64, filenameMap[presetType]);
      setScanResult(result);
    } catch (err: any) {
      setScanError(err.message || 'Failed to analyze imaging scan');
    } finally {
      setIsScanning(false);
    }
  };

  const proceedWithScan = () => {
    if (!scanResult) return;
    if (scanResult.breast_cancer_diagnosis) {
      navigate('/common/breast_cancer');
      return;
    }
    const findingsText = scanResult.findings
      .map(f => `${f.label} (${f.hpo_id})`)
      .join(', ');
    const fullText = `Patient imaging scan (${scanResult.modality_detected}): ${findingsText}. ${scanResult.findings.map(f => f.clinical_evidence).join(' ')}`;
    navigate('/case/new/nlp', { state: { rawText: fullText } });
  };

  return (
    <div className="min-h-screen bg-bg">
      {/* Header */}
      <header className="border-b border-border bg-surface">
        <div className="max-w-5xl mx-auto px-8 py-6 flex flex-col sm:flex-row justify-between items-start sm:items-end gap-3">
          <div>
            <h1 className="text-2xl font-semibold text-ink tracking-tight">QResolve</h1>
            <p className="text-sm text-ink-muted mt-1">Hybrid Quantum-Classical Diagnostic Assistant • ABDM Compliant</p>
          </div>
          <div className="flex items-center gap-2">
            <a
              href={api.getDemoReportPdfUrl()}
              target="_blank"
              rel="noopener noreferrer"
              download="QResolve_Clinical_Report_DEMO.pdf"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 border border-border bg-gray-50 hover:bg-gray-100 text-ink text-xs font-medium rounded transition-colors shadow-sm"
              title="Download sample ReportLab clinical differential report"
            >
              <span>📄</span>
              <span>Sample PDF Report</span>
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-8 py-10">
        {/* Hidden file input */}
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept="image/*,.dcm,.dicom"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleScanUpload(e.target.files[0]);
            }
          }}
        />

        {/* Intro */}
        <div className="mb-8">
          <h2 className="text-lg font-semibold text-ink mb-2">Benchmark Cases & Diagnostics</h2>
          <p className="text-sm text-ink-muted leading-relaxed max-w-2xl">
            Select a disease category or upload radiological imaging (X-Ray / MRI / Echo) to walk through the full diagnostic pipeline.
            <strong className="text-ink"> Classical ML</strong> resolves routine presentations directly.
            <strong className="text-primary"> Rare disease</strong> cases with overlapping phenotypes escalate to the Quantum SVM resolver.
          </p>
        </div>

        {/* Multi-Modal Imaging Scan Uploader Card */}
        <div className="mb-10 bg-surface border border-border p-6 shadow-sm">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 pb-4 border-b border-border">
            <div>
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-quantum inline-block"></span>
                <h3 className="font-semibold text-base text-ink">Multi-Modal Imaging (Computer Vision Module)</h3>
                <Badge variant="primary" className="text-[10px]">NEW</Badge>
              </div>
              <p className="text-xs text-ink-muted mt-1">
                Upload DICOM, X-Ray, Echocardiogram, or MRA scans to automatically extract phenotype biomarkers into the symptom vector.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                disabled={isScanning}
                className="text-xs border-quantum/60 text-quantum-text hover:bg-quantum-soft/20"
              >
                {isScanning ? (
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-3 border-2 border-quantum border-t-transparent rounded-full animate-spin"></span>
                    Analyzing Radiomics...
                  </span>
                ) : (
                  '📁 Upload Scan (X-Ray / Echo / MRI)'
                )}
              </Button>

              <div className="flex items-center gap-1 border-l border-border pl-2">
                <span className="text-[10px] text-ink-muted font-mono uppercase mr-1">Presets:</span>
                <button
                  onClick={() => handlePresetScan('chest_xray')}
                  disabled={isScanning}
                  className="text-[11px] text-ink-muted hover:text-primary px-2 py-1 bg-gray-50 hover:bg-gray-100 rounded border border-border/60"
                  title="Test Chest X-Ray radiomics"
                >
                  + Chest X-Ray
                </button>
                <button
                  onClick={() => handlePresetScan('echo')}
                  disabled={isScanning}
                  className="text-[11px] text-ink-muted hover:text-primary px-2 py-1 bg-gray-50 hover:bg-gray-100 rounded border border-border/60"
                  title="Test Echocardiogram radiomics"
                >
                  + Echo
                </button>
                <button
                  onClick={() => handlePresetScan('mra')}
                  disabled={isScanning}
                  className="text-[11px] text-ink-muted hover:text-primary px-2 py-1 bg-gray-50 hover:bg-gray-100 rounded border border-border/60"
                  title="Test MR Angiogram radiomics"
                >
                  + MRA
                </button>
                <button
                  onClick={() => handlePresetScan('mammogram')}
                  disabled={isScanning}
                  className="text-[11px] text-blue-700 bg-blue-50 hover:bg-blue-100 px-2 py-1 rounded border border-blue-200"
                  title="Test Digital Mammogram (Breast Cancer)"
                >
                  + Mammogram (Breast)
                </button>
              </div>
            </div>
          </div>

          {scanError && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 text-red-700 text-xs">
              Imaging Scan Error: {scanError}
            </div>
          )}

          {scanResult && (
            <div className="mt-4 p-4 bg-quantum-soft/20 border border-quantum/30 rounded-none animate-fade-in">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 mb-3">
                <div>
                  <span className="text-xs font-semibold uppercase text-quantum-text tracking-wide">
                    Radiological Biomarkers Detected: {scanResult.modality_detected}
                  </span>
                  <div className="text-[11px] text-ink-muted">
                    Quantitative radiomics extracted {scanResult.detected_hpo_ids.length} phenotypic markers mapped to HPO ontology.
                  </div>
                </div>
                <Button onClick={proceedWithScan} className="text-xs py-1.5 px-3">
                  Triage Patient with Detected Symptoms →
                </Button>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {scanResult.findings.map((f, i) => (
                  <div key={i} className="bg-surface p-3 border border-border/80 text-xs">
                    <div className="flex justify-between items-start mb-1">
                      <span className="font-semibold text-ink">{f.label}</span>
                      <Badge variant="primary" className="text-[10px]">
                        {(f.confidence * 100).toFixed(0)}% Conf
                      </Badge>
                    </div>
                    <div className="text-[11px] text-quantum-text font-mono mb-1">{f.hpo_id} • {f.anatomical_region}</div>
                    <div className="text-[11px] text-ink-muted leading-relaxed">{f.clinical_evidence}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {loading && <div className="text-ink-muted">Loading available diseases from backend...</div>}
        {error && <div className="text-red-600">Error: {error}</div>}

        {!loading && !error && (
          <>
            {/* Common Diseases Section */}
            <section className="mb-10">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                <h3 className="text-sm font-semibold uppercase tracking-wide text-blue-600">Common Diseases — Classical ML</h3>
                <div className="flex-1 h-px bg-border"></div>
                <span className="text-[11px] text-ink-muted font-mono">STANDALONE PIPELINE</span>
              </div>

              <div className="grid grid-cols-2 gap-4">
                {diseases.filter(d => d.category === 'common').map(disease => (
                  <button
                    key={disease.id}
                    onClick={() => navigate(`/common/${disease.id}`)}
                    className="group text-left bg-surface border border-border p-5 hover:border-blue-500 hover:shadow-[0_0_0_1px_var(--color-blue-500)] transition-all duration-150"
                  >
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <div className="font-semibold text-ink text-[15px] group-hover:text-blue-600 transition-colors">{disease.name}</div>
                        <div className="text-xs text-ink-muted mt-0.5">{disease.description}</div>
                      </div>
                      <div className="flex flex-col items-end gap-1">
                        <span className="text-xs font-mono px-2 py-0.5 bg-blue-50 text-blue-600 font-medium">Common</span>
                      </div>
                    </div>
                    <p className="text-sm text-ink-muted font-serif leading-relaxed">Features: {disease.n_symptoms}</p>
                    <div className="mt-3 pt-3 border-t border-border flex items-center gap-2 text-xs text-ink-muted">
                      <span className="w-1.5 h-1.5 rounded-full bg-blue-500 inline-block"></span>
                      Classical ML (Sklearn / XGBoost)
                    </div>
                  </button>
                ))}
              </div>
            </section>

            {/* Rare Diseases Section */}
            <section>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-3 h-3 rounded-full bg-quantum"></div>
                <h3 className="text-sm font-semibold uppercase tracking-wide text-quantum-text">Rare Diseases — Hybrid Pipeline</h3>
                <div className="flex-1 h-px bg-border"></div>
                <span className="text-[11px] text-ink-muted font-mono">5-DISEASE CLUSTER</span>
              </div>

              <div className="grid grid-cols-3 gap-4">
                {diseases.filter(d => d.category === 'rare').map(disease => (
                  <button
                    key={disease.id}
                    onClick={() => navigate(`/case/${disease.id}/input`)}
                    className={`group text-left bg-surface border border-border p-5 transition-all duration-150 ${disease.track === 'quantum' ? 'hover:border-quantum hover:shadow-[0_0_0_1px_var(--color-quantum)]' : 'hover:border-primary hover:shadow-[0_0_0_1px_var(--color-primary)]'}`}
                  >
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <div className={`font-semibold text-[15px] transition-colors ${disease.track === 'quantum' ? 'text-ink group-hover:text-quantum-text' : 'text-ink group-hover:text-primary'}`}>{disease.name}</div>
                        <div className="text-xs text-ink-muted mt-0.5">{disease.description}</div>
                      </div>
                      <span className={`text-xs font-mono px-2 py-0.5 font-medium ${disease.track === 'quantum' ? 'bg-quantum-soft text-quantum-text' : 'bg-primary-soft text-primary'}`}>
                        {disease.track === 'quantum' ? 'Quantum' : 'Classical'}
                      </span>
                    </div>
                    <p className="text-sm text-ink-muted font-serif leading-relaxed">Known HPO Symptoms: {disease.n_symptoms}</p>
                    <div className="mt-3 pt-3 border-t border-border flex justify-between items-center text-xs text-ink-muted">
                      <div className="flex items-center gap-2">
                        <span className={`w-1.5 h-1.5 rounded-full inline-block ${disease.track === 'quantum' ? 'bg-quantum' : 'bg-primary'}`}></span>
                        {disease.track === 'quantum' ? 'Classical → QSVM → XAI' : 'XGBoost → XAI'}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
};
