"use client";

import { useEffect, useRef, useState } from "react";
import { Activity, AlertCircle, BarChart2, Brain, CheckCircle2, Download, FileText, History, RotateCcw, Upload, Zap } from "lucide-react";
import jsPDF from "jspdf";
import { checkApiHealth, predictImage, type PredictionResponse } from "../lib/api";

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const CLASS_LABELS: Record<string, string> = { glioma: "Glioma", meningioma: "Meningioma", notumor: "No tumor", pituitary: "Pituitary" };
const CLASS_COLORS: Record<string, string> = { glioma: "#d84444", meningioma: "#6750a4", notumor: "#16866a", pituitary: "#b7791f" };

type ScanRecord = { id: string; filename: string; timestamp: string; result: PredictionResponse };
type Tab = "scan" | "history" | "analytics";

function classLabel(value: string) { return CLASS_LABELS[value] ?? value; }

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [history, setHistory] = useState<ScanRecord[]>(() => {
    if (typeof window === "undefined") return [];
    try {
      return JSON.parse(localStorage.getItem("brain_tumor_classification_history") ?? "[]");
    } catch {
      return [];
    }
  });
  const [activeImage, setActiveImage] = useState<keyof PredictionResponse["images"]>("overlay");
  const [tab, setTab] = useState<Tab>("scan");
  const [loading, setLoading] = useState(false);
  const [serviceOnline, setServiceOnline] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    checkApiHealth().then(() => setServiceOnline(true)).catch(() => setServiceOnline(false));
  }, []);

  const selectFile = (nextFile: File) => {
    if (!nextFile.type.startsWith("image/")) return setError("Choose a JPG or PNG image.");
    if (nextFile.size > MAX_FILE_BYTES) return setError("Choose an image smaller than 10 MB.");
    setFile(nextFile); setResult(null); setError(null);
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result as string);
    reader.readAsDataURL(nextFile);
  };

  const reset = () => { setFile(null); setPreview(null); setResult(null); setError(null); };

  const analyse = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    try {
      const response = await predictImage(file);
      setResult(response); setServiceOnline(true);
      const updated = [{ id: crypto.randomUUID(), filename: file.name, timestamp: new Date().toLocaleString(), result: response }, ...history].slice(0, 50);
      setHistory(updated);
      localStorage.setItem("brain_tumor_classification_history", JSON.stringify(updated));
    } catch (caught) {
      setServiceOnline(false);
      setError(caught instanceof Error ? caught.message : "The analysis service is unavailable.");
    } finally { setLoading(false); }
  };

  const downloadReport = (scan: PredictionResponse, filename: string) => {
    const doc = new jsPDF();
    doc.setFontSize(19); doc.setFont("helvetica", "bold"); doc.text("Brain Tumor Classification — Research Report", 14, 20);
    doc.setFontSize(10); doc.setFont("helvetica", "normal"); doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 28); doc.text(`Image: ${filename}`, 14, 34);
    doc.setDrawColor(210, 214, 224); doc.line(14, 39, 196, 39);
    doc.setFontSize(13); doc.setFont("helvetica", "bold"); doc.text("Classification result", 14, 50);
    doc.setFontSize(11); doc.setFont("helvetica", "normal");
    doc.text(`Predicted class: ${classLabel(scan.prediction.class)}`, 14, 59);
    doc.text(`Model confidence: ${scan.prediction.confidence.toFixed(1)}%`, 14, 66);
    doc.setFontSize(9); doc.setTextColor(90, 90, 100); doc.text("For research and educational use only. This output is not a medical diagnosis.", 14, 76);
    doc.setTextColor(0, 0, 0); doc.setFontSize(13); doc.setFont("helvetica", "bold"); doc.text("Class confidence scores", 14, 91);
    let y = 100; doc.setFontSize(10); doc.setFont("helvetica", "normal");
    Object.entries(scan.prediction.confidences).sort(([, a], [, b]) => b - a).forEach(([name, score]) => { doc.text(`${classLabel(name)}: ${score.toFixed(1)}%`, 14, y); y += 7; });
    doc.setFontSize(13); doc.setFont("helvetica", "bold"); doc.text("Grad-CAM overlay", 14, y + 11);
    doc.addImage(scan.images.overlay, "PNG", 14, y + 16, 90, 90);
    doc.save(`brain-tumor-classification-${scan.prediction.class}.pdf`);
  };

  const stats = { total: history.length, tumor: history.filter((entry) => entry.result.prediction.class !== "notumor").length, clear: history.filter((entry) => entry.result.prediction.class === "notumor").length };

  return <main className="app-shell">
    <header className="topbar">
      <div className="brand"><span className="brand-mark"><Brain size={21} /></span><div><strong>Brain Tumor Classification</strong><small>Research MRI image analysis</small></div></div>
      <nav aria-label="Application views">{([ ["scan", "New scan", Zap], ["history", `History (${history.length})`, History], ["analytics", "Analytics", BarChart2] ] as const).map(([id, label, Icon]) => <button key={id} className={tab === id ? "nav-button active" : "nav-button"} onClick={() => setTab(id)}><Icon size={15} />{label}</button>)}</nav>
      <div className="service-status"><i className={serviceOnline === false ? "offline" : ""} />{serviceOnline === null ? "Checking service" : serviceOnline ? "Service online" : "Service unavailable"}</div>
    </header>

    <section className="page-content">
      {tab === "scan" && <div className={result ? "scan-grid with-result" : "scan-grid"}>
        <div className="stack">
          <section className="card"><p className="eyebrow">Upload MRI image</p>
            {!preview ? <button className={dragging ? "drop-zone dragging" : "drop-zone"} onClick={() => inputRef.current?.click()} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); const dropped = event.dataTransfer.files.item(0); if (dropped) selectFile(dropped); }}><Upload size={32} /><strong>Drop an MRI image here</strong><span>JPG or PNG · maximum 10 MB</span></button> : <><img className="preview" src={preview} alt="Selected MRI preview" /><p className="file-detail">{file?.name} · {Math.round((file?.size ?? 0) / 1024)} KB</p></>}
            <input ref={inputRef} className="sr-only" type="file" accept="image/png,image/jpeg" onChange={(event) => { const selected = event.target.files?.item(0); if (selected) selectFile(selected); }} />
          </section>
          {preview && <div className="actions"><button className="primary-action" disabled={loading} onClick={analyse}>{loading ? <><Activity className="spin" size={17} />Analysing image…</> : <><Zap size={17} />Analyse image</>}</button><button className="secondary-icon" onClick={reset} aria-label="Choose another image"><RotateCcw size={18} /></button></div>}
          {error && <div className="error-message"><AlertCircle size={18} />{error}</div>}
          <section className="card"><p className="eyebrow">Supported research classes</p><div className="class-list">{Object.entries(CLASS_LABELS).map(([id, label]) => <span key={id}><i style={{ backgroundColor: CLASS_COLORS[id] }} />{label}</span>)}</div></section>
        </div>
        {result && <div className="stack"><section className="card result-card"><div className="result-title"><CheckCircle2 size={21} style={{ color: CLASS_COLORS[result.prediction.class] }} /><p className="eyebrow">Classification result</p></div><div className="metric"><strong style={{ color: CLASS_COLORS[result.prediction.class] }}>{classLabel(result.prediction.class)}</strong><span>predicted research class</span></div><div className="confidence-list">{Object.entries(result.prediction.confidences).sort(([, a], [, b]) => b - a).map(([name, score]) => <div key={name}><p><span>{classLabel(name)}</span><strong>{score.toFixed(1)}%</strong></p><div className="progress"><i style={{ width: `${score}%`, backgroundColor: CLASS_COLORS[name] }} /></div></div>)}</div><p className="disclaimer">For research and educational use only. This output is not a medical diagnosis.</p><button className="report-button" onClick={() => downloadReport(result, file?.name ?? "scan")}><FileText size={16} />Download research report</button></section>
          <section className="card"><p className="eyebrow">Grad-CAM visualization</p><div className="image-tabs">{(["overlay", "heatmap", "original"] as const).map((image) => <button key={image} className={activeImage === image ? "selected" : ""} onClick={() => setActiveImage(image)}>{image}</button>)}</div><img className="heatmap" src={result.images[activeImage]} alt={`${activeImage} visualization`} /><p className="help-text">Highlighted regions indicate areas that influenced this model output.</p></section></div>}
      </div>}

      {tab === "history" && <section><h1>Local scan history</h1><p className="page-intro">Saved only in this browser. No images are stored by the application server.</p><div className="history-list">{history.length ? history.map((entry) => <article className="history-card" key={entry.id}><div><strong>{entry.filename}</strong><small>{entry.timestamp}</small></div><div><span style={{ color: CLASS_COLORS[entry.result.prediction.class] }}>{classLabel(entry.result.prediction.class)}</span><small>{entry.result.prediction.confidence.toFixed(1)}% confidence</small></div><button className="report-button" onClick={() => downloadReport(entry.result, entry.filename)}><Download size={15} />Report</button></article>) : <div className="empty-state">No local scan history yet.</div>}</div></section>}

      {tab === "analytics" && <section><h1>Local analytics</h1><p className="page-intro">These counts describe scans stored in this browser only; they are not clinical statistics.</p><div className="stat-grid"><div className="stat"><strong>{stats.total}</strong><span>Total scans</span></div><div className="stat"><strong>{stats.tumor}</strong><span>Non-&quot;no tumor&quot; outputs</span></div><div className="stat"><strong>{stats.clear}</strong><span>&quot;No tumor&quot; outputs</span></div></div></section>}
    </section>
  </main>;
}
