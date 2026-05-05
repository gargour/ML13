import { BrowserRouter, Routes, Route } from "react-router-dom";

import Dashboard from "./pages/Dashboard";
import Training from "./pages/Training";
import Results from "./pages/Results";
import Dataset from "./pages/Dataset";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/training" element={<Training />} />
        <Route path="/results" element={<Results />} />
        <Route path="/dataset" element={<Dataset />} />
      </Routes>
    </BrowserRouter>
  );
}