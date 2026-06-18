"use client";

import { useState, useRef, useEffect } from "react";
import {
  Upload, Brain, Activity, AlertCircle,
  CheckCircle, Download, RotateCcw, Zap,
  History, BarChart2, X, FileText
} from "lucide-react";
import jsPDF from "jspdf";

const CLASS_COLORS: Record<string, string> = {
  glioma: "#E24B4A",
  meningioma: "#534AB7",
  notumor: "#1D9E75",
  pituitary: "#BA7517",
};

const CLASS_LABELS: Record<string, string> = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  notumor: "No Tumor",
  pituitary: "Pituitary",
};

type ScanRecord = {
  id: string;
  filename: string;
  timestamp: string;
  predicted: string;
  confidence: number;
  images: { original: string; heatmap: string; overlay: string };
  confidences: Record<string, number>;
};

type Tab = "scan" | "history" | "analytics";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [activeImg, setActiveImg] = useState<"original" | "heatmap" | "overlay">("overlay");
  const [tab, setTab] = useState<Tab>("scan");
  const [history, setHistory] = useState<ScanRecord[]>([]);
  const [selectedHistory, setSelectedHistory] = useState<ScanRecord | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const saved = localStorage.getItem("neuroscan_history");
    if (saved) setHistory(JSON.parse(saved));
  }, []);

  const saveToHistory = (res: any, fname: string) => {
    const record: ScanRecord = {
      id: Date.now().toString(),
      filename: fname,
      timestamp: new Date().toLocaleString(),
      predicted: res.prediction.class,
      confidence: res.prediction.confidence,
      images: res.images,
      confidences: res.prediction.confidences,
    };
    const updated = [record, ...history].slice(0, 50);
    setHistory(updated);
    localStorage.setItem("neuroscan_history", JSON.stringify(updated));
  };

  const handleFile = (f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) handleFile(f);
  };

  const handleAnalyse = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("https://neuroscan-ai-production-1cfe.up.railway.app/api/predict", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Server error");
      const data = await res.json();
      setResult(data);
      saveToHistory(data, file.name);
    } catch {
      setError("Could not connect to AI server. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const downloadPDF = async (res: any, fname: string) => {
    const doc = new jsPDF();
    const predicted = res.prediction.class;
    const confidence = res.prediction.confidence;

    doc.setFillColor(15, 15, 19);
    doc.rect(0, 0, 210, 297, "F");

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(20);
    doc.setFont("helvetica", "bold");
    doc.text("NeuroScan AI — Scan Report", 14, 20);

    doc.setFontSize(10);
    doc.setTextColor(150, 150, 150);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 28);
    doc.text(`File: ${fname}`, 14, 34);

    doc.setDrawColor(40, 40, 60);
    doc.line(14, 38, 196, 38);

    doc.setFontSize(13);
    doc.setTextColor(255, 255, 255);
    doc.setFont("helvetica", "bold");
    doc.text("Prediction", 14, 48);

    doc.setFontSize(11);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(180, 180, 180);
    doc.text(`Detected class: ${CLASS_LABELS[predicted]}`, 14, 57);
    doc.text(`Confidence: ${confidence.toFixed(1)}%`, 14, 64);
    doc.text(`Severity: ${res.prediction.info.severity}`, 14, 71);
    doc.text(`Description: ${res.prediction.info.description}`, 14, 78, { maxWidth: 182 });

    doc.setFontSize(13);
    doc.setTextColor(255, 255, 255);
    doc.setFont("helvetica", "bold");
    doc.text("Confidence Scores", 14, 95);

    let y = 104;
    Object.entries(res.prediction.confidences)
      .sort(([, a]: any, [, b]: any) => b - a)
      .forEach(([cls, conf]: any) => {
        doc.setFontSize(10);
        doc.setFont("helvetica", "normal");
        doc.setTextColor(180, 180, 180);
        doc.text(`${CLASS_LABELS[cls]}: ${conf.toFixed(1)}%`, 14, y);
        y += 7;
      });

    if (res.images?.overlay) {
      doc.setFontSize(13);
      doc.setTextColor(255, 255, 255);
      doc.setFont("helvetica", "bold");
      doc.text("Grad-CAM Heatmap (Overlay)", 14, y + 8);
      doc.addImage(res.images.overlay, "PNG", 14, y + 14, 80, 80);
    }

    doc.setFontSize(9);
    doc.setTextColor(80, 80, 80);
    doc.text("For research purposes only. Not a medical diagnosis.", 14, 285);

    doc.save(`neuroscan_${predicted}_report.pdf`);
  };

  const predicted = result?.prediction?.class;
  const confidence = result?.prediction?.confidence;
  const confidences = result?.prediction?.confidences || {};

  const analytics = {
    total: history.length,
    tumor: history.filter(h => h.predicted !== "notumor").length,
    clear: history.filter(h => h.predicted === "notumor").length,
    avgConf: history.length
      ? (history.reduce((s, h) => s + h.confidence, 0) / history.length).toFixed(1)
      : "0",
    classCounts: Object.keys(CLASS_LABELS).map(cls => ({
      cls,
      count: history.filter(h => h.predicted === cls).length
    }))
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0f0f13" }}>

      {/* Top bar */}
      <div style={{
        background: "#16161f", borderBottom: "1px solid #1e1e2e",
        padding: "14px 32px", display: "flex",
        alignItems: "center", justifyContent: "space-between",
        position: "sticky", top: 0, zIndex: 50
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: "#1e1a4a",
            display: "flex", alignItems: "center", justifyContent: "center"
          }}>
            <Brain size={20} color="#7c6fff" />
          </div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: "#fff" }}>NeuroScan AI</div>
            <div style={{ fontSize: 11, color: "#666" }}>Brain tumor detection</div>
          </div>
        </div>

        {/* Nav tabs */}
        <div style={{ display: "flex", gap: 4 }}>
          {([
            { id: "scan", icon: <Zap size={14} />, label: "New scan" },
            { id: "history", icon: <History size={14} />, label: `History (${history.length})` },
            { id: "analytics", icon: <BarChart2 size={14} />, label: "Analytics" },
          ] as any[]).map(t => (
            <button key={t.id} onClick={() => setTab(t.id)}
              style={{
                display: "flex", alignItems: "center", gap: 6,
                padding: "7px 14px", borderRadius: 8, border: "none",
                background: tab === t.id ? "#1e1a4a" : "transparent",
                color: tab === t.id ? "#7c6fff" : "#666",
                fontSize: 13, cursor: "pointer", fontWeight: tab === t.id ? 500 : 400
              }}>
              {t.icon}{t.label}
            </button>
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#1D9E75" }} />
          <span style={{ fontSize: 12, color: "#1D9E75" }}>EfficientNet-B0 · 95.19% acc</span>
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 24px" }}>

        {/* ── SCAN TAB ── */}
        {tab === "scan" && (
          <div style={{
            display: "grid",
            gridTemplateColumns: result ? "1fr 1fr" : "1fr",
            gap: 20
          }}>
            {/* Left panel */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div style={{
                background: "#16161f", border: "1px solid #1e1e2e",
                borderRadius: 16, padding: 24
              }}>
                <div style={{ fontSize: 12, fontWeight: 500, color: "#555", marginBottom: 14, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  Upload MRI scan
                </div>

                {!preview ? (
                  <div
                    className={`upload-zone ${dragging ? "dragging" : ""}`}
                    style={{ padding: "48px 24px", textAlign: "center", cursor: "pointer" }}
                    onClick={() => fileRef.current?.click()}
                    onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={handleDrop}
                  >
                    <Upload size={32} color="#444" style={{ margin: "0 auto 12px" }} />
                    <div style={{ fontSize: 14, color: "#aaa", marginBottom: 4 }}>Drop your MRI image here</div>
                    <div style={{ fontSize: 12, color: "#555" }}>Supports JPG, PNG · Max 10MB</div>
                    <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
                      onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
                  </div>
                ) : (
                  <div>
                    <img src={preview} alt="MRI preview" style={{
                      width: "100%", borderRadius: 10,
                      border: "1px solid #1e1e2e",
                      maxHeight: 280, objectFit: "contain", background: "#0a0a0f"
                    }} />
                    <div style={{ marginTop: 10, fontSize: 12, color: "#666" }}>
                      {file?.name} · {((file?.size || 0) / 1024).toFixed(0)} KB
                    </div>
                  </div>
                )}
              </div>

              {preview && (
                <div style={{ display: "flex", gap: 10 }}>
                  <button onClick={handleAnalyse} disabled={loading} style={{
                    flex: 1, padding: "12px 0",
                    background: loading ? "#2a2a3a" : "#534AB7",
                    color: "#fff", border: "none", borderRadius: 10,
                    fontSize: 14, fontWeight: 500, cursor: loading ? "not-allowed" : "pointer",
                    display: "flex", alignItems: "center", justifyContent: "center", gap: 8
                  }}>
                    {loading ? <><Activity size={16} />Analysing...</> : <><Zap size={16} />Analyse MRI</>}
                  </button>
                  <button onClick={reset} style={{
                    padding: "12px 16px", background: "transparent",
                    color: "#888", border: "1px solid #1e1e2e",
                    borderRadius: 10, cursor: "pointer"
                  }}>
                    <RotateCcw size={16} />
                  </button>
                </div>
              )}

              {error && (
                <div style={{
                  background: "#2a1515", border: "1px solid #5a2020",
                  borderRadius: 10, padding: 14,
                  display: "flex", gap: 10, alignItems: "flex-start"
                }}>
                  <AlertCircle size={16} color="#E24B4A" style={{ marginTop: 1, flexShrink: 0 }} />
                  <span style={{ fontSize: 13, color: "#E24B4A" }}>{error}</span>
                </div>
              )}

              {!preview && (
                <div style={{
                  background: "#16161f", border: "1px solid #1e1e2e",
                  borderRadius: 16, padding: 20
                }}>
                  <div style={{ fontSize: 12, fontWeight: 500, color: "#555", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                    Detectable tumor types
                  </div>
                  {Object.entries(CLASS_LABELS).map(([key, label]) => (
                    <div key={key} style={{
                      display: "flex", alignItems: "center", gap: 10,
                      padding: "8px 0", borderBottom: "1px solid #1e1e2e"
                    }}>
                      <div style={{ width: 10, height: 10, borderRadius: "50%", background: CLASS_COLORS[key], flexShrink: 0 }} />
                      <span style={{ fontSize: 13, color: "#ccc" }}>{label}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Right panel */}
            {result && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div style={{
                  background: "#16161f",
                  border: `1px solid ${CLASS_COLORS[predicted] || "#1e1e2e"}`,
                  borderRadius: 16, padding: 24
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                    <CheckCircle size={20} color={CLASS_COLORS[predicted]} />
                    <div style={{ fontSize: 12, fontWeight: 500, color: "#555", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                      Prediction result
                    </div>
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 20 }}>
                    {[
                      { val: CLASS_LABELS[predicted], label: "Detected", color: CLASS_COLORS[predicted] },
                      { val: `${confidence?.toFixed(1)}%`, label: "Confidence", color: "#fff" },
                      { val: result?.prediction?.info?.severity === "none" ? "Clear" : result?.prediction?.info?.severity === "medium" ? "Medium" : "High", label: "Severity", color: "#fff" }
                    ].map((s, i) => (
                      <div key={i} style={{ background: "#0f0f13", borderRadius: 10, padding: 12, textAlign: "center" }}>
                        <div style={{ fontSize: 18, fontWeight: 600, color: s.color }}>{s.val}</div>
                        <div style={{ fontSize: 11, color: "#555", marginTop: 3 }}>{s.label}</div>
                      </div>
                    ))}
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {Object.entries(confidences)
                      .sort(([, a]: any, [, b]: any) => b - a)
                      .map(([cls, conf]: any) => (
                        <div key={cls}>
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#888", marginBottom: 4 }}>
                            <span>{CLASS_LABELS[cls]}</span>
                            <span style={{ fontWeight: 500, color: "#ccc" }}>{conf.toFixed(1)}%</span>
                          </div>
                          <div style={{ height: 6, background: "#0f0f13", borderRadius: 3, overflow: "hidden" }}>
                            <div style={{
                              height: "100%", width: `${conf}%`,
                              background: CLASS_COLORS[cls], borderRadius: 3,
                              transition: "width 0.8s ease"
                            }} />
                          </div>
                        </div>
                      ))}
                  </div>

                  <div style={{ marginTop: 16, padding: 12, background: "#0f0f13", borderRadius: 10 }}>
                    <div style={{ fontSize: 12, color: "#888", lineHeight: 1.6 }}>
                      {result?.prediction?.info?.description}
                    </div>
                  </div>

                  {/* Action buttons */}
                  <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
                    <button onClick={() => downloadPDF(result, file?.name || "scan")}
                      style={{
                        display: "flex", alignItems: "center", gap: 6,
                        padding: "8px 14px", background: "#1e1a4a",
                        border: "1px solid #534AB7", borderRadius: 8,
                        color: "#7c6fff", fontSize: 12, cursor: "pointer"
                      }}>
                      <FileText size={13} />Download PDF
                    </button>
                    <button onClick={() => {
                      const a = document.createElement("a");
                      a.href = result?.images?.overlay;
                      a.download = `neuroscan_${predicted}_heatmap.png`;
                      a.click();
                    }} style={{
                      display: "flex", alignItems: "center", gap: 6,
                      padding: "8px 14px", background: "transparent",
                      border: "1px solid #1e1e2e", borderRadius: 8,
                      color: "#888", fontSize: 12, cursor: "pointer"
                    }}>
                      <Download size={13} />Save heatmap
                    </button>
                  </div>
                </div>

                {/* Heatmap viewer */}
                <div style={{
                  background: "#16161f", border: "1px solid #1e1e2e",
                  borderRadius: 16, padding: 24
                }}>
                  <div style={{ fontSize: 12, fontWeight: 500, color: "#555", marginBottom: 14, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                    Grad-CAM heatmap
                  </div>
                  <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
                    {(["overlay", "heatmap", "original"] as const).map(t => (
                      <button key={t} onClick={() => setActiveImg(t)} style={{
                        padding: "5px 12px", borderRadius: 20, border: "1px solid",
                        borderColor: activeImg === t ? "#534AB7" : "#1e1e2e",
                        background: activeImg === t ? "#1e1a4a" : "transparent",
                        color: activeImg === t ? "#7c6fff" : "#666",
                        fontSize: 12, cursor: "pointer", textTransform: "capitalize"
                      }}>{t}</button>
                    ))}
                  </div>
                  <img src={result?.images?.[activeImg]} alt={activeImg} style={{
                    width: "100%", borderRadius: 10,
                    border: "1px solid #1e1e2e",
                    maxHeight: 280, objectFit: "contain", background: "#0a0a0f"
                  }} />
                  <div style={{ marginTop: 10, fontSize: 11, color: "#555" }}>
                    Red areas = where the AI focused most to make this prediction
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── HISTORY TAB ── */}
        {tab === "history" && (
          <div>
            <div style={{ fontSize: 18, fontWeight: 600, color: "#fff", marginBottom: 20 }}>
              Scan history
            </div>
            {history.length === 0 ? (
              <div style={{
                background: "#16161f", border: "1px solid #1e1e2e",
                borderRadius: 16, padding: 48, textAlign: "center"
              }}>
                <History size={32} color="#333" style={{ margin: "0 auto 12px" }} />
                <div style={{ color: "#555", fontSize: 14 }}>No scans yet. Analyse an MRI to see history.</div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {history.map(scan => (
                  <div key={scan.id}
                    onClick={() => setSelectedHistory(selectedHistory?.id === scan.id ? null : scan)}
                    style={{
                      background: "#16161f",
                      border: `1px solid ${selectedHistory?.id === scan.id ? CLASS_COLORS[scan.predicted] : "#1e1e2e"}`,
                      borderRadius: 12, padding: 16,
                      cursor: "pointer", transition: "border-color 0.2s"
                    }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <div style={{
                        width: 10, height: 10, borderRadius: "50%",
                        background: CLASS_COLORS[scan.predicted], flexShrink: 0
                      }} />
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 13, fontWeight: 500, color: "#fff" }}>{scan.filename}</div>
                        <div style={{ fontSize: 11, color: "#555", marginTop: 2 }}>{scan.timestamp}</div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: 13, fontWeight: 500, color: CLASS_COLORS[scan.predicted] }}>
                          {CLASS_LABELS[scan.predicted]}
                        </div>
                        <div style={{ fontSize: 11, color: "#555" }}>{scan.confidence.toFixed(1)}% confidence</div>
                      </div>
                    </div>

                    {selectedHistory?.id === scan.id && (
                      <div style={{ marginTop: 14, paddingTop: 14, borderTop: "1px solid #1e1e2e" }}>
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 12 }}>
                          {(["original", "heatmap", "overlay"] as const).map(t => (
                            <div key={t}>
                              <img src={scan.images[t]} alt={t} style={{
                                width: "100%", borderRadius: 8,
                                border: "1px solid #1e1e2e"
                              }} />
                              <div style={{ fontSize: 10, color: "#555", textAlign: "center", marginTop: 4, textTransform: "capitalize" }}>{t}</div>
                            </div>
                          ))}
                        </div>
                        <button onClick={(e) => { e.stopPropagation(); downloadPDF({ prediction: { class: scan.predicted, confidence: scan.confidence, confidences: scan.confidences, info: { severity: "medium", description: "" } }, images: scan.images }, scan.filename); }}
                          style={{
                            display: "flex", alignItems: "center", gap: 6,
                            padding: "7px 12px", background: "#1e1a4a",
                            border: "1px solid #534AB7", borderRadius: 8,
                            color: "#7c6fff", fontSize: 12, cursor: "pointer"
                          }}>
                          <FileText size={13} />Download PDF report
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── ANALYTICS TAB ── */}
        {tab === "analytics" && (
          <div>
            <div style={{ fontSize: 18, fontWeight: 600, color: "#fff", marginBottom: 20 }}>
              Analytics
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 20 }}>
              {[
                { val: analytics.total, label: "Total scans", color: "#fff" },
                { val: analytics.tumor, label: "Tumor detected", color: "#E24B4A" },
                { val: analytics.clear, label: "Clear scans", color: "#1D9E75" },
                { val: `${analytics.avgConf}%`, label: "Avg confidence", color: "#534AB7" },
              ].map((s, i) => (
                <div key={i} style={{
                  background: "#16161f", border: "1px solid #1e1e2e",
                  borderRadius: 14, padding: 20, textAlign: "center"
                }}>
                  <div style={{ fontSize: 28, fontWeight: 600, color: s.color }}>{s.val}</div>
                  <div style={{ fontSize: 12, color: "#555", marginTop: 4 }}>{s.label}</div>
                </div>
              ))}
            </div>

            <div style={{
              background: "#16161f", border: "1px solid #1e1e2e",
              borderRadius: 16, padding: 24
            }}>
              <div style={{ fontSize: 13, fontWeight: 500, color: "#555", marginBottom: 16, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                Class distribution
              </div>
              {analytics.total === 0 ? (
                <div style={{ color: "#555", fontSize: 13, textAlign: "center", padding: 24 }}>
                  No data yet. Analyse some MRI scans first.
                </div>
              ) : (
                analytics.classCounts.map(({ cls, count }) => (
                  <div key={cls} style={{ marginBottom: 14 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#888", marginBottom: 6 }}>
                      <span>{CLASS_LABELS[cls]}</span>
                      <span style={{ color: "#ccc" }}>{count} scans · {analytics.total ? ((count / analytics.total) * 100).toFixed(0) : 0}%</span>
                    </div>
                    <div style={{ height: 8, background: "#0f0f13", borderRadius: 4, overflow: "hidden" }}>
                      <div style={{
                        height: "100%",
                        width: analytics.total ? `${(count / analytics.total) * 100}%` : "0%",
                        background: CLASS_COLORS[cls], borderRadius: 4,
                        transition: "width 0.8s ease"
                      }} />
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}