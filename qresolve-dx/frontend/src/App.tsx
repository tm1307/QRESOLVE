import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { MainLayout } from './components/layout/MainLayout';
import { DoctorDashboard } from './components/dashboard/DoctorDashboard';
import { PatientInput } from './components/stages/1_PatientInput';
import { NLPProcessing } from './components/stages/2_NLPProcessing';
import { KnowledgeGraph } from './components/stages/3_KnowledgeGraph';
import { ClassicalTriage } from './components/stages/4_ClassicalTriage';
import { ConfusionDetection } from './components/stages/5_ConfusionDetection';
import { QuantumResolver } from './components/stages/6_QuantumResolver';
import { ExplainableOutput } from './components/stages/7_ExplainableOutput';
import { NextTestRecommendation } from './components/stages/8_NextTestRecommendation';
import { CommonDiseaseInput } from './components/stages/CommonDiseaseInput';


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DoctorDashboard />} />
        
        <Route path="/common/breast-cancer" element={<Navigate to="/common/breast_cancer" replace />} />
        <Route path="/common/breast_cancer" element={
          <div className="min-h-screen bg-bg">
            <header className="border-b border-border bg-surface px-8 py-6 flex justify-between items-center">
              <h1 className="text-xl font-semibold text-ink cursor-pointer" onClick={() => window.location.href='/'}>QResolve</h1>
            </header>
            <CommonDiseaseInput diseaseType="breast-cancer" />
          </div>
        } />
        
        <Route path="/common/parkinsons" element={
          <div className="min-h-screen bg-bg">
            <header className="border-b border-border bg-surface px-8 py-6 flex justify-between items-center">
              <h1 className="text-xl font-semibold text-ink cursor-pointer" onClick={() => window.location.href='/'}>QResolve</h1>
            </header>
            <CommonDiseaseInput diseaseType="parkinsons" />
          </div>
        } />

        <Route path="/case/:id" element={<MainLayout />}>
          <Route path="input" element={<PatientInput />} />
          <Route path="nlp" element={<NLPProcessing />} />
          <Route path="graph" element={<KnowledgeGraph />} />
          <Route path="triage" element={<ClassicalTriage />} />
          <Route path="confusion" element={<ConfusionDetection />} />
          <Route path="quantum" element={<QuantumResolver />} />
          <Route path="explain" element={<ExplainableOutput />} />
          <Route path="test" element={<NextTestRecommendation />} />
          <Route index element={<Navigate to="input" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
