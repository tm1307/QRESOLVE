import { useLocation, useNavigate } from 'react-router-dom';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';

export const QuantumResolver = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const triageResult = location.state?.triageResult;

  if (!triageResult) return null;

  const metrics = triageResult.circuit_metrics;

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl">
      <div>
        <h2 className="text-xl font-semibold text-ink flex items-center gap-3">
          6. Quantum Resolver
          <Badge variant="quantum">Resolved</Badge>
        </h2>
        <p className="text-ink-muted text-sm mt-1">Quantum Support Vector Machine (QSVM) classification complete.</p>
      </div>

      <Card className="p-0 border-t-4 border-quantum overflow-hidden shadow-[0_8px_30px_rgba(14,165,160,0.12)]">
        <div className="bg-quantum-soft border-b border-border px-6 py-4 flex justify-between items-center">
          <h3 className="text-sm font-semibold text-quantum-text uppercase tracking-wide">Quantum Margin Separation</h3>
          <span className="text-xs font-mono text-quantum-text/80">
            Kernel: {metrics?.kernel_type || 'ZZFeatureMap'} ({metrics?.n_qubits || 8} qubits)
          </span>
        </div>
        
        <div className="p-10 text-center bg-surface">
          <div className="text-xs text-ink-muted uppercase tracking-wider font-semibold mb-3">Final Conclusion</div>
          <div className="text-4xl font-semibold text-quantum-text mb-4">{triageResult.top_diagnosis}</div>
          <div className="text-[15px] font-mono text-ink-muted flex justify-center items-center gap-4">
            <span className="bg-gray-50 px-3 py-1 border border-border">Confidence: {(triageResult.confidence * 100).toFixed(1)}%</span>
            <span>•</span>
            <span className="bg-gray-50 px-3 py-1 border border-border">Status: {triageResult.quantum_status}</span>
          </div>
        </div>

        {/* Circuit metrics telemetry — only shown when real quantum computation ran */}
        {metrics && (
          <div className="border-t border-border px-6 py-4 bg-quantum-soft/30">
            <div className="text-xs text-ink-muted uppercase tracking-wider font-semibold mb-3">Circuit Telemetry</div>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-semibold text-quantum-text">{metrics.n_qubits}</div>
                <div className="text-xs text-ink-muted mt-1">Qubits</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-semibold text-quantum-text">{metrics.reps}</div>
                <div className="text-xs text-ink-muted mt-1">Circuit Reps</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-semibold text-quantum-text">{metrics.computation_time_s}s</div>
                <div className="text-xs text-ink-muted mt-1">Kernel Time</div>
              </div>
            </div>
            <div className="mt-3 text-center text-xs text-ink-muted">
              Entanglement: {metrics.entanglement} • {metrics.pre_trained ? 'Pre-trained QSVM' : 'Live-trained QSVM'}
            </div>
          </div>
        )}
      </Card>

      <div className="flex justify-between items-center pt-4">
        <Button variant="outline" onClick={() => navigate(-1)}>← Back</Button>
        <Button variant="quantum" onClick={() => navigate(`/case/${triageResult.case_id}/explain`, { state: { triageResult } })}>
          Generate Explanation →
        </Button>
      </div>
    </div>
  );
};
