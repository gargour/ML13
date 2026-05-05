import { useEffect, useState } from "react";

export default function ResultsPage() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchResults = async () => {
    try {
      setLoading(true);

      const res = await fetch("http://127.0.0.1:5000/mlflow-runs");
      const data = await res.json();

      setRuns(data);
    } catch (err) {
      console.log("Error fetching results:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResults();
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <h1>📊 MLflow Results Dashboard</h1>

      <button onClick={fetchResults}>
        🔄 Refresh
      </button>

      {loading && <p>Loading results...</p>}

      {!loading && runs.length === 0 && (
        <p>No results found. Run training first.</p>
      )}

      <div style={{ marginTop: 20 }}>
        {runs.map((run, i) => (
          <div
            key={i}
            style={{
              border: "1px solid #ccc",
              padding: 15,
              marginBottom: 10,
              borderRadius: 8
            }}
          >
            <h3>🧠 Model: {run.model}</h3>
            <p>🎯 Accuracy: {run.accuracy}</p>
            <p>📊 F1 Score: {run.f1}</p>
            <p>📌 Precision: {run.precision}</p>
            <p>📌 Recall: {run.recall}</p>
          </div>
        ))}
      </div>
    </div>
  );
}