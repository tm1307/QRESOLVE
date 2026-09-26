import { useEffect, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { api } from '../../lib/api';
import type { ExplainResponse } from '../../types';

export const NextTestRecommendation = () => {
  const { id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [explainData, setExplainData] = useState<ExplainResponse | null>(location.state?.explainData || null);
  const [isLoading, setIsLoading] = useState(!location.state?.explainData);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (explainData) return;
    if (!id || id === 'new') {
      setIsLoading(false);
      return;
    }

    const fetchExplain = async () => {
      try {
        const data = await api.explain(id);
        setExplainData(data);
      } catch (err: any) {
        setError(err.message || 'Failed to fetch test recommendations');
      } finally {
        setIsLoading(false);
      }
    };

    fetchExplain();
  }, [id, explainData]);

  if (isLoading) return <div className="p-8 text-ink-muted font-serif">Loading test recommendations...</div>;
  if (error) return <div className="p-8 text-red-600 bg-red-50">Error: {error}</div>;
  if (!explainData) return <div className="p-8">No test recommendations available. <Button onClick={() => navigate('/')}>Return to Dashboard</Button></div>;

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl">
      <div>
        <h2 className="text-xl font-semibold text-ink">8. Next Test Recommendation</h2>
        <p className="text-ink-muted text-sm mt-1">Suggested clinical tests optimized for Bayesian Information Gain to finalize diagnosis.</p>
      </div>

      <div className="space-y-4">
        {explainData.suggested_tests.map((test: any, idx: number) => (
          <Card key={idx} className={`p-6 ${idx === 0 ? 'border-l-4 border-l-primary bg-primary-soft/10 shadow-sm' : 'opacity-80'}`}>
            <div className="flex flex-col md:flex-row md:justify-between items-start gap-4">
              <div className="flex-1">
                <div className="text-xs text-primary uppercase tracking-wider font-semibold mb-2">
                  {idx === 0 ? 'Highest Info Gain' : `Alternative ${idx}`}
                </div>
                <h3 className="font-semibold text-ink text-lg">{test.clinical_test || test.label}</h3>
                <p className="text-ink-muted text-[15px] mt-2 font-serif">To confirm presence/absence of: <strong>{test.label}</strong></p>
                
                <div className="mt-4 space-y-2 text-sm text-ink-muted bg-white p-3 border border-border">
                  <p><span className="font-semibold text-green-700">If Positive:</span> {test.expected_outcome_positive} becomes more likely.</p>
                  <p><span className="font-semibold text-red-700">If Negative:</span> {test.expected_outcome_negative} becomes more likely.</p>
                </div>
              </div>
              <div className="text-right shrink-0 min-w-[120px]">
                <div className="text-xs text-ink-muted uppercase font-semibold mb-1">Gain</div>
                <div className="font-mono text-primary font-bold bg-white border border-border px-3 py-2 text-lg inline-block">
                  {test.information_gain.toFixed(3)}
                </div>
                {/* Visual bar for info gain */}
                <div className="w-full bg-gray-200 h-1.5 mt-2 rounded-full overflow-hidden">
                  <div className="bg-primary h-full rounded-full" style={{ width: `${Math.min(100, (test.information_gain / 0.5) * 100)}%` }}></div>
                </div>
              </div>
            </div>
          </Card>
        ))}
        
        {explainData.suggested_tests.length === 0 && (
          <Card className="p-8 text-center bg-gray-50 border-dashed border-2">
            <p className="text-ink-muted font-serif">No further tests mathematically recommended for this specific differential.</p>
          </Card>
        )}
      </div>

      <div className="flex justify-between items-center pt-4">
        <Button variant="outline" onClick={() => navigate(-1)}>← Back</Button>
        <Button onClick={() => navigate('/')}>Complete Case & Return to Dashboard</Button>
      </div>
    </div>
  );
};
