import { useEffect, useState } from 'react';
import * as d3 from 'd3-force';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { useNavigate, useParams, useLocation } from 'react-router-dom';
import { api } from '../../lib/api';
import type { GraphNode, GraphData } from '../../types';

export const KnowledgeGraph = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { id } = useParams();
  
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [links, setLinks] = useState<any[]>([]);
  
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [isGenerating, setIsGenerating] = useState(true);
  const [isTriaging, setIsTriaging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchGraph = async () => {
      try {
        const confirmedTerms = location.state?.confirmedTerms || [];
        if (confirmedTerms.length === 0) {
          setIsGenerating(false);
          return;
        }
        
        const data = await api.getGraph(confirmedTerms);
        setGraphData(data);
      } catch (err: any) {
        setError(err.message || 'Failed to fetch graph data');
      } finally {
        setIsGenerating(false);
      }
    };
    
    fetchGraph();
  }, [location.state?.confirmedTerms]);

  useEffect(() => {
    if (isGenerating || !graphData) return;
    
    // Copy data so D3 can mutate the positions
    const nodesCopy = graphData.nodes.map(d => ({ ...d, x: undefined, y: undefined }));
    const linksCopy = graphData.links.map(d => ({ ...d }));

    const simulation = d3.forceSimulation(nodesCopy as any)
      .force('link', d3.forceLink(linksCopy).id((d: any) => d.id).distance(130))
      .force('charge', d3.forceManyBody().strength(-650))
      .force('center', d3.forceCenter(400, 250))
      .force('collision', d3.forceCollide().radius((d: any) => {
        // ERR-07 fix: collision radius based on node type + label padding
        return d.group === 'disease' ? 40 : 28;
      }).strength(0.9))
      .on('tick', () => {
        setNodes([...nodesCopy as any]);
        setLinks([...linksCopy]);
      });

    return () => { simulation.stop(); };
  }, [graphData, isGenerating]);

  const handleRunTriage = async () => {
    const confirmedTerms = location.state?.confirmedTerms || [];
    setIsTriaging(true);
    setError(null);
    try {
      const response = await api.diagnose({ symptoms: confirmedTerms, case_id: id });
      navigate(`/case/${response.case_id}/triage`, { state: { triageResult: response } });
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsTriaging(false);
    }
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl">
      <div>
        <h2 className="text-xl font-semibold text-ink">3. Knowledge Graph Overview</h2>
        <p className="text-ink-muted text-sm mt-1">Headless D3 layout mapping symptoms to disease clusters in the QResolve database.</p>
      </div>

      <Card className="p-0 overflow-hidden bg-bg border-border relative h-[500px]">
        {isGenerating ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-surface/80 z-20">
            <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4"></div>
            <p className="text-ink font-medium">Querying Knowledge Graph...</p>
            <p className="text-ink-muted text-sm mt-1">Generating unified disease-symptom graph</p>
          </div>
        ) : null}

        {/* Legend */}
        <div className={`absolute top-4 right-4 flex gap-4 bg-surface/90 border border-border px-4 py-2 z-10 transition-opacity ${isGenerating ? 'opacity-0' : 'opacity-100'}`}>
          <div className="flex items-center gap-2 text-xs">
            <span className="w-3 h-3 rounded-full bg-primary inline-block"></span>
            <span className="text-ink-muted">Disease</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="w-3 h-3 rounded-full bg-quantum inline-block"></span>
            <span className="text-ink-muted">Symptom</span>
          </div>
        </div>

        <svg width="100%" height="100%" viewBox="0 0 800 500" className="w-full h-full">
          {/* Render Links */}
          <g stroke="#D0D5DD" strokeWidth="1.5" strokeDasharray="4 3">
            {links.map((link, i) => (
              <line 
                key={i} 
                x1={link.source.x} 
                y1={link.source.y} 
                x2={link.target.x} 
                y2={link.target.y} 
              />
            ))}
          </g>
          {/* Render Nodes */}
          <g>
            {nodes.map((node: any, i) => {
              const isDisease = node.group === 'disease';
              return (
                <g key={i} transform={`translate(${node.x || 0},${node.y || 0})`} className="cursor-pointer transition-transform hover:scale-110">
                  <circle 
                    r={isDisease ? 14 : 9} 
                    fill={isDisease ? '#3457D5' : '#0EA5A0'} 
                    stroke={isDisease ? '#EAF0FD' : '#E3F6F4'}
                    strokeWidth="4"
                  />
                  <text 
                    dy={isDisease ? 28 : 22} 
                    textAnchor="middle" 
                    fill={isDisease ? '#1B2430' : '#5B6472'} 
                    fontSize={isDisease ? 13 : 11}
                    fontWeight={isDisease ? 600 : 500}
                    fontFamily="IBM Plex Sans, sans-serif"
                    className="select-none"
                  >
                    {node.label}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      </Card>

      {error && (
        <Card className="p-4 border-l-4 border-l-red-500 bg-red-50 text-sm text-red-800">
          <strong>FastAPI Error:</strong> {error}
        </Card>
      )}

      <div className="flex justify-between items-center pt-4">
        <Button variant="outline" onClick={() => navigate(-1)}>← Back to NLP</Button>
        <Button onClick={handleRunTriage} disabled={isGenerating || isTriaging || (nodes.length === 0)}>
          {isTriaging ? 'Running Triage...' : 'Run Classical Triage →'}
        </Button>
      </div>
    </div>
  );
};
