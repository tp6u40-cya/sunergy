// src/pages/PredictSolar.js
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import Navbar from '../components/Navbar';
import * as XLSX from 'xlsx';
import { saveAs } from 'file-saver';

/* ── 誤差燈號組件 ── */
const ErrorLight = ({ pct }) => {
  if (pct === null || pct === undefined) return <span className="text-white/20 text-[10px]">—</span>;
  const v = Number(pct);
  if (v <= 5) return (
    <span className="inline-flex items-center gap-1.5">
      <span className="size-2.5 rounded-full bg-green-400 shadow-[0_0_6px_rgba(34,197,94,0.5)]" />
      <span className="text-green-400 text-[10px] font-mono">{v.toFixed(1)}%</span>
    </span>
  );
  if (v <= 15) return (
    <span className="inline-flex items-center gap-1.5">
      <span className="size-2.5 rounded-full bg-yellow-400 shadow-[0_0_6px_rgba(234,179,8,0.5)]" />
      <span className="text-yellow-400 text-[10px] font-mono">{v.toFixed(1)}%</span>
    </span>
  );
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="size-2.5 rounded-full bg-red-400 shadow-[0_0_6px_rgba(239,68,68,0.5)] animate-pulse" />
      <span className="text-red-400 text-[10px] font-mono">{v.toFixed(1)}%</span>
    </span>
  );
};

/* ── 整體診斷燈號 ── */
const OverallStatus = ({ avgError, label }) => {
  if (avgError === null || avgError === undefined) return null;
  const v = Number(avgError);
  let cfg = { color: 'text-green-400', bg: 'bg-green-500', shadow: 'shadow-[0_0_15px_rgba(34,197,94,0.4)]', label: '發電正常', desc: '預測與實際高度吻合' };
  if (v > 15) cfg = { color: 'text-red-400', bg: 'bg-red-500', shadow: 'shadow-[0_0_15px_rgba(239,68,68,0.4)]', label: '發電異常', desc: '偏差過大，請檢查設備' };
  else if (v > 5) cfg = { color: 'text-yellow-400', bg: 'bg-yellow-500', shadow: 'shadow-[0_0_15px_rgba(234,179,8,0.4)]', label: '需留意', desc: '環境干擾或輕微積塵' };

  return (
    <div className="flex items-center gap-3">
      <div className="relative flex items-center justify-center">
        <div className={`size-8 rounded-full ${cfg.bg} ${cfg.shadow} animate-pulse`} />
      </div>
      <div className="flex flex-col">
        {label && <p className="text-[9px] text-white/30 font-bold uppercase tracking-widest">{label}</p>}
        <span className={`text-sm font-black ${cfg.color}`}>{cfg.label}</span>
        <span className="text-[9px] text-white/30 italic">{cfg.desc}</span>
      </div>
    </div>
  );
};

/* ── 模型顏色表 ── */
const MODEL_COLORS = [
  { text: 'text-primary',    bg: 'bg-primary/15',    border: 'border-primary/30',    dot: 'bg-primary' },
  { text: 'text-cyan-400',   bg: 'bg-cyan-400/15',   border: 'border-cyan-400/30',   dot: 'bg-cyan-400' },
  { text: 'text-violet-400', bg: 'bg-violet-400/15', border: 'border-violet-400/30', dot: 'bg-violet-400' },
  { text: 'text-rose-400',   bg: 'bg-rose-400/15',   border: 'border-rose-400/30',   dot: 'bg-rose-400' },
  { text: 'text-emerald-400',bg: 'bg-emerald-400/15',border: 'border-emerald-400/30',dot: 'bg-emerald-400' },
  { text: 'text-orange-400', bg: 'bg-orange-400/15', border: 'border-orange-400/30', dot: 'bg-orange-400' },
];

/* ── 排序方向圖示 ── */
const SortIcon = ({ direction }) => {
  if (!direction) return <span className="text-white/15 text-[8px] ml-0.5">⇅</span>;
  return <span className="text-primary text-[9px] ml-0.5 font-bold">{direction === 'asc' ? '↑' : '↓'}</span>;
};

/* ── 取得燈號顏色 hex ── */
const getErrorColor = (pct) => {
  if (pct === null || pct === undefined) return null;
  const v = Number(pct);
  if (isNaN(v)) return null;
  if (v <= 5) return { bg: 'C6EFCE', fg: '006100' };   // 綠
  if (v <= 15) return { bg: 'FFEB9C', fg: '9C6500' };  // 黃
  return { bg: 'FFC7CE', fg: '9C0006' };                // 紅
};

export default function PredictSolar({ onBack, onNavigateToDashboard, onLogout, onNavigateToSites, onNavigateToTrain, onNavigateToPredict, onNavigateToModelMgmt }) {
  const [file, setFile] = useState(null);
  const [selectedModelIds, setSelectedModelIds] = useState([]);

  useEffect(() => {
    const modelId = localStorage.getItem("predict_model_id");
    if (modelId) {
      setSelectedModelIds([modelId]);
      localStorage.removeItem("predict_model_id");
    }
  }, []);

  const [trainedModels, setTrainedModels] = useState([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // pagination
  const PAGE_SIZE = 20;
  const [page, setPage] = useState(0);

  // sorting state: { key: string, direction: 'asc' | 'desc' | null }
  const [sortConfig, setSortConfig] = useState({ key: null, direction: null });

  // filter state: { [colName]: { operator: '>' | '<' | '=' | '>=' | '<=' | 'contains', value: string } }
  const [filters, setFilters] = useState({});
  const [showFilters, setShowFilters] = useState(false);

  // Fetch trained models on mount
  useEffect(() => {
    const user = JSON.parse(localStorage.getItem("user"));
    if (!user || !user.user_id) {
      setError("找不到登入資訊，請重新登入");
      return;
    }
    const userId = user.user_id;
    fetch(`http://127.0.0.1:8000/train/trained-models?user_id=${userId}`)
      .then(async (r) => {
        const data = await r.json().catch(() => []);
        if (!r.ok) throw new Error(data?.detail || '取得模型失敗');
        return data;
      })
      .then((data) => {
        if (Array.isArray(data)) setTrainedModels(data);
      })
      .catch((e) => setError(e.message || '取得模型失敗'));
  }, []);

  const toggleModel = (modelId) => {
    const id = String(modelId);
    setSelectedModelIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  const handlePredict = async () => {
    if (!file || selectedModelIds.length === 0) return alert('請上傳資料並選擇至少一個模型');
    setIsPredicting(true);
    setError('');
    setResult(null);
    setSortConfig({ key: null, direction: null });
    setFilters({});
    try {
      const formData = new FormData();
      formData.append('file', file);

      if (selectedModelIds.length === 1) {
        formData.append('model_id', selectedModelIds[0]);
        const res = await fetch('http://127.0.0.1:8000/train/predict-file', {
          method: 'POST',
          body: formData,
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json?.detail || '預測失敗');
        setResult({
          mode: 'single',
          models_summary: [{
            model_id: json.model_id,
            model_type: json.model_type,
            status: 'ok',
            total_predicted_eac: json.total_predicted_eac,
            avg_error_pct: json.avg_error_pct,
          }],
          total_rows: json.total_rows,
          columns: json.columns,
          rows: json.rows,
        });
      } else {
        formData.append('model_ids', selectedModelIds.join(','));
        const res = await fetch('http://127.0.0.1:8000/train/predict-file-multi', {
          method: 'POST',
          body: formData,
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json?.detail || '預測失敗');
        setResult({ mode: 'multi', ...json });
      }
      setPage(0);
    } catch (e) {
      setError(e.message || '預測過程發生錯誤');
    } finally {
      setIsPredicting(false);
    }
  };

  const navProps = { onNavigateToDashboard, onNavigateToTrain, onNavigateToPredict, onNavigateToSites, onNavigateToModelMgmt, onLogout };

  const displayCols = result ? result.columns : [];

  // Detect which columns are prediction/error columns
  const isPredCol = (col) => col.startsWith('pred_') || col === 'predicted_EAC';
  const isErrCol = (col) => col.startsWith('err_') || col === 'error_pct';

  // Determine if a column is numeric based on first few non-null values
  const isNumericCol = useCallback((col) => {
    if (!result?.rows) return false;
    for (const row of result.rows.slice(0, 20)) {
      const v = row[col];
      if (v !== null && v !== undefined && v !== '' && v !== '—') {
        return typeof v === 'number' || (!isNaN(Number(v)) && v !== '');
      }
    }
    return false;
  }, [result]);

  // Get model info from column name like "pred_XGBoost_5"
  const getModelIdFromCol = (col) => {
    const parts = col.split('_');
    return parts.length >= 3 ? parseInt(parts[parts.length - 1]) : null;
  };

  // Build color map for selected models
  const okModels = (result?.models_summary || []).filter(m => m.status === 'ok');
  const modelColorMap = {};
  okModels.forEach((m, i) => {
    modelColorMap[m.model_id] = MODEL_COLORS[i % MODEL_COLORS.length];
  });

  // ── Filtering logic ──
  const filteredRows = useMemo(() => {
    if (!result?.rows) return [];
    const allRows = result.rows;
    const activeFilters = Object.entries(filters).filter(([, f]) => f.value !== '' && f.value !== undefined);
    if (activeFilters.length === 0) return allRows;

    return allRows.filter(row => {
      return activeFilters.every(([col, filter]) => {
        const cellVal = row[col];
        if (cellVal === null || cellVal === undefined) return false;

        if (filter.operator === 'contains') {
          return String(cellVal).toLowerCase().includes(String(filter.value).toLowerCase());
        }

        const numVal = Number(cellVal);
        const filterNum = Number(filter.value);
        if (isNaN(numVal) || isNaN(filterNum)) return false;

        switch (filter.operator) {
          case '>':  return numVal > filterNum;
          case '<':  return numVal < filterNum;
          case '=':  return Math.abs(numVal - filterNum) < 0.0001;
          case '>=': return numVal >= filterNum;
          case '<=': return numVal <= filterNum;
          default:   return true;
        }
      });
    });
  }, [result, filters]);

  // ── Sorting logic ──
  const sortedRows = useMemo(() => {
    if (!sortConfig.key || !sortConfig.direction) return filteredRows;

    return [...filteredRows].sort((a, b) => {
      const aVal = a[sortConfig.key];
      const bVal = b[sortConfig.key];

      // Handle nulls: push to end
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;

      const aNum = Number(aVal);
      const bNum = Number(bVal);

      let comparison = 0;
      if (!isNaN(aNum) && !isNaN(bNum)) {
        comparison = aNum - bNum;
      } else {
        comparison = String(aVal).localeCompare(String(bVal), 'zh-Hant');
      }

      return sortConfig.direction === 'asc' ? comparison : -comparison;
    });
  }, [filteredRows, sortConfig]);

  // ── Pagination on sorted+filtered data ──
  const totalPages = Math.ceil(sortedRows.length / PAGE_SIZE);
  const pagedRows = sortedRows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  // ── Sort handler ──
  const handleSort = (colKey) => {
    setSortConfig(prev => {
      if (prev.key !== colKey) return { key: colKey, direction: 'asc' };
      if (prev.direction === 'asc') return { key: colKey, direction: 'desc' };
      return { key: null, direction: null };
    });
    setPage(0);
  };

  // ── Filter handler ──
  const updateFilter = (col, field, value) => {
    setFilters(prev => ({
      ...prev,
      [col]: { ...prev[col], [field]: value },
    }));
    setPage(0);
  };

  const clearAllFilters = () => {
    setFilters({});
    setPage(0);
  };

  const hasActiveFilters = Object.values(filters).some(f => f.value !== '' && f.value !== undefined);

  // ── Clean column label ──
  const getColLabel = (col) => {
    if (col === 'predicted_EAC') return '預測 EAC';
    if (col === 'error_pct') return '誤差%';
    if (col.startsWith('pred_')) {
      const parts = col.replace('pred_', '').split('_');
      return `預測 ${parts.slice(0, -1).join('_')}`;
    }
    if (col.startsWith('err_')) {
      const parts = col.replace('err_', '').split('_');
      return `誤差% ${parts.slice(0, -1).join('_')}`;
    }
    return col;
  };

  // ── XLSX Download ──
  const handleDownloadXlsx = () => {
    if (!result) return;
    const dataToExport = sortedRows;

    // Build header row: # + displayCols + (single mode: 燈號)
    const headerLabels = ['#', ...displayCols.map(getColLabel)];
    if (result.mode === 'single') headerLabels.push('燈號');

    // Build data rows
    const wsData = [headerLabels];
    dataToExport.forEach((row, idx) => {
      const rowArr = [idx + 1];
      displayCols.forEach(col => {
        const val = row[col];
        rowArr.push(val === null || val === undefined ? '' : val);
      });
      if (result.mode === 'single') {
        const ep = row.error_pct;
        if (ep !== null && ep !== undefined) {
          const v = Number(ep);
          rowArr.push(v <= 5 ? '正常' : v <= 15 ? '留意' : '異常');
        } else {
          rowArr.push('');
        }
      }
      wsData.push(rowArr);
    });

    // Create worksheet
    const ws = XLSX.utils.aoa_to_sheet(wsData);

    // Apply styles to error columns (cell background colors)
    // Find which column indices are error columns
    const errColIndices = [];
    displayCols.forEach((col, ci) => {
      if (isErrCol(col)) {
        errColIndices.push(ci + 1); // +1 because col 0 is '#'
      }
    });

    // Also check single mode 燈號 column
    const lightColIdx = result.mode === 'single' ? headerLabels.length - 1 : -1;

    // Apply cell styles for each data row
    for (let r = 1; r < wsData.length; r++) {
      // Style error percentage columns
      errColIndices.forEach(c => {
        const cellAddr = XLSX.utils.encode_cell({ r, c });
        const cellVal = wsData[r][c];
        const colorInfo = getErrorColor(cellVal);
        if (colorInfo && ws[cellAddr]) {
          ws[cellAddr].s = {
            fill: { fgColor: { rgb: colorInfo.bg } },
            font: { color: { rgb: colorInfo.fg }, bold: true },
          };
        }
      });

      // Style 燈號 column for single mode
      if (lightColIdx >= 0) {
        const cellAddr = XLSX.utils.encode_cell({ r, c: lightColIdx });
        const errPctColIdx = displayCols.indexOf('error_pct');
        const errVal = errPctColIdx >= 0 ? wsData[r][errPctColIdx + 1] : null;
        const colorInfo = getErrorColor(errVal);
        if (colorInfo && ws[cellAddr]) {
          ws[cellAddr].s = {
            fill: { fgColor: { rgb: colorInfo.bg } },
            font: { color: { rgb: colorInfo.fg }, bold: true },
            alignment: { horizontal: 'center' },
          };
        }
      }
    }

    // Style header row
    for (let c = 0; c < headerLabels.length; c++) {
      const cellAddr = XLSX.utils.encode_cell({ r: 0, c });
      if (ws[cellAddr]) {
        ws[cellAddr].s = {
          fill: { fgColor: { rgb: '1F2937' } },
          font: { color: { rgb: 'FFFFFF' }, bold: true },
          alignment: { horizontal: 'center' },
        };
      }
    }

    // Set column widths
    ws['!cols'] = headerLabels.map((h, i) => ({
      wch: i === 0 ? 5 : Math.max(h.length * 2.5, 12),
    }));

    // Create workbook and save
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, '預測結果');

    const now = new Date();
    const ts = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`;
    const filename = `prediction_result_${ts}.xlsx`;

    const wbout = XLSX.write(wb, { bookType: 'xlsx', type: 'array', cellStyles: true });
    saveAs(new Blob([wbout], { type: 'application/octet-stream' }), filename);
  };

  return (
    <div className="min-h-screen w-full bg-background-dark text-white flex flex-col font-sans">
      <Navbar activePage="predict" {...navProps} />

      <main className="flex-1 w-full max-w-[1400px] mx-auto p-6 py-10 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">

        {/* ── 左側：配置區 ── */}
        <div className="lg:col-span-3 flex flex-col gap-6">
          <section className="bg-white/[0.02] p-6 rounded-2xl border border-white/10 shadow-2xl">
            <h2 className="text-lg font-bold text-white mb-6 flex items-center gap-3 italic">
              <div className="size-8 rounded-lg bg-primary/20 text-primary flex items-center justify-center">
                <span className="material-symbols-outlined !text-xl">settings_applications</span>
              </div>
              預測配置
            </h2>

            <div className="space-y-6">
              {/* 1. Upload */}
              <div>
                <label className="text-[11px] text-white/30 mb-2 block font-bold uppercase tracking-widest">1. 上傳預測資料</label>
                <div
                  className="group border-2 border-dashed border-white/10 rounded-2xl p-6 text-center hover:bg-white/[0.03] hover:border-primary/50 transition-all cursor-pointer"
                  onClick={() => document.getElementById('predictFileInput').click()}
                >
                  <input type="file" id="predictFileInput" hidden accept=".csv,.xlsx,.xls" onChange={(e) => setFile(e.target.files[0])} />
                  <div className="size-10 rounded-full bg-white/5 mx-auto mb-3 flex items-center justify-center group-hover:bg-primary/10 transition-colors">
                    <span className="material-symbols-outlined !text-xl text-white/30 group-hover:text-primary">upload_file</span>
                  </div>
                  <p className="text-xs font-bold text-white/50 group-hover:text-white transition-colors">{file ? file.name : 'CSV / XLSX 檔案'}</p>
                </div>
              </div>

              {/* 2. Multi-Model Select */}
              <div>
                <label className="text-[11px] text-white/30 mb-2 block font-bold uppercase tracking-widest">
                  2. 選擇訓練模型
                  <span className="ml-2 text-primary/60 normal-case">（可多選比對）</span>
                </label>
                <div className="max-h-[280px] overflow-y-auto space-y-1.5 pr-1 custom-scrollbar">
                  {trainedModels.length === 0 && (
                    <p className="text-[10px] text-white/20 italic py-4 text-center">尚無可用模型</p>
                  )}
                  {trainedModels.map(m => {
                    const isSelected = selectedModelIds.includes(String(m.model_id));
                    return (
                      <label
                        key={m.model_id}
                        className={`flex items-center gap-3 px-3 py-2.5 rounded-xl cursor-pointer transition-all border ${
                          isSelected
                            ? 'bg-primary/10 border-primary/30 shadow-[0_0_12px_rgba(242,204,13,0.08)]'
                            : 'border-white/5 hover:bg-white/[0.03] hover:border-white/10'
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleModel(m.model_id)}
                          className="accent-primary size-3.5 rounded"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className={`text-[10px] font-black px-1.5 py-0.5 rounded ${
                              isSelected ? 'bg-primary/20 text-primary' : 'bg-white/5 text-white/50'
                            }`}>
                              {m.model_type}
                            </span>
                            <span className="text-[10px] text-white/30 font-mono">#{m.model_id}</span>
                          </div>
                          <p className="text-[9px] text-white/20 mt-0.5 truncate">
                            {m.site_name} · {m.trained_at ? m.trained_at.slice(0, 16).replace('T', ' ') : ''}
                          </p>
                        </div>
                      </label>
                    );
                  })}
                </div>
                {selectedModelIds.length > 0 && (
                  <p className="text-[10px] text-primary/60 mt-2 font-bold">
                    已選 {selectedModelIds.length} 個模型
                  </p>
                )}
              </div>
            </div>

            <button
              onClick={handlePredict}
              disabled={isPredicting || !file || selectedModelIds.length === 0}
              className="w-full bg-primary text-background-dark py-4 rounded-2xl font-black text-sm hover:scale-[1.02] active:scale-95 transition-all shadow-[0_10px_30px_rgba(242,204,13,0.2)] mt-6 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
            >
              {isPredicting ? '運算執行中...' : selectedModelIds.length > 1 ? `比對預測 (${selectedModelIds.length} 個模型)` : '開始執行預測'}
            </button>

            {error && (
              <div className="mt-4 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-xs">
                {error}
              </div>
            )}
          </section>

          {/* Summary cards — show per-model comparison */}
          {result && (
            <section className="space-y-3 animate-fade-in">
              {okModels.map((m, i) => {
                const c = MODEL_COLORS[i % MODEL_COLORS.length];
                return (
                  <div key={m.model_id} className={`${c.bg} border ${c.border} p-4 rounded-2xl`}>
                    <div className="flex items-center gap-2 mb-3">
                      <span className={`size-2.5 rounded-full ${c.dot}`} />
                      <span className={`text-xs font-black ${c.text}`}>{m.model_type}</span>
                      <span className="text-[9px] text-white/30 font-mono">#{m.model_id}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <p className="text-[9px] text-white/30 font-bold uppercase tracking-widest mb-1">預估發電量</p>
                        <div className="flex items-baseline gap-1">
                          <span className="text-lg font-black font-mono text-white">{m.total_predicted_eac?.toLocaleString() ?? '—'}</span>
                          <span className={`text-[10px] font-bold ${c.text}`}>kWh</span>
                        </div>
                      </div>
                      <div>
                        <p className="text-[9px] text-white/30 font-bold uppercase tracking-widest mb-1">平均誤差</p>
                        <div className="flex items-baseline gap-1">
                          <span className="text-lg font-black font-mono text-white">{m.avg_error_pct !== null && m.avg_error_pct !== undefined ? m.avg_error_pct.toFixed(2) : '—'}</span>
                          <span className={`text-[10px] font-bold ${c.text}`}>%</span>
                        </div>
                      </div>
                    </div>
                    <div className="mt-3 pt-3 border-t border-white/5">
                      <OverallStatus avgError={m.avg_error_pct} />
                    </div>
                  </div>
                );
              })}
            </section>
          )}
        </div>

        {/* ── 右側：資料表格 ── */}
        <div className="lg:col-span-9 flex flex-col">
          <div className={`flex-1 w-full rounded-2xl border border-white/10 bg-white/[0.01] p-6 flex flex-col relative transition-all shadow-2xl ${!result && 'items-center justify-center border-dashed opacity-40 min-h-[500px]'}`}>
            {result ? (
              <div className="w-full flex flex-col animate-fade-in">
                {/* Header with actions */}
                <div className="flex flex-wrap justify-between items-center mb-4 gap-3">
                  <h2 className="text-sm font-bold text-white flex items-center gap-2">
                    <span className="material-symbols-outlined !text-base text-primary">table_chart</span>
                    預測結果
                    <span className="text-white/30 font-normal ml-2">
                      {hasActiveFilters
                        ? `篩選後 ${sortedRows.length} / 共 ${result.total_rows} 筆`
                        : `共 ${result.total_rows} 筆`
                      }
                    </span>
                  </h2>

                  <div className="flex items-center gap-2">
                    {/* Model legend for multi-mode */}
                    {okModels.length > 1 && (
                      <div className="flex items-center gap-3 mr-2">
                        {okModels.map((m, i) => {
                          const c = MODEL_COLORS[i % MODEL_COLORS.length];
                          return (
                            <span key={m.model_id} className="flex items-center gap-1.5">
                              <span className={`size-2 rounded-full ${c.dot}`} />
                              <span className={`text-[10px] font-bold ${c.text}`}>{m.model_type}#{m.model_id}</span>
                            </span>
                          );
                        })}
                      </div>
                    )}
                    {okModels.length === 1 && (
                      <div className="text-[10px] text-white/30 mr-2">
                        模型：<span className="text-primary font-bold">{okModels[0].model_type}</span>
                      </div>
                    )}

                    {/* Toggle filter button */}
                    <button
                      onClick={() => setShowFilters(f => !f)}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[10px] font-bold border transition-all ${
                        showFilters
                          ? 'bg-primary/10 border-primary/30 text-primary'
                          : 'border-white/10 text-white/40 hover:bg-white/5 hover:text-white/60'
                      }`}
                    >
                      <span className="material-symbols-outlined !text-sm">filter_alt</span>
                      篩選
                      {hasActiveFilters && (
                        <span className="size-4 rounded-full bg-primary text-background-dark text-[8px] font-black flex items-center justify-center">
                          {Object.values(filters).filter(f => f.value).length}
                        </span>
                      )}
                    </button>

                    {/* Clear filters */}
                    {hasActiveFilters && (
                      <button
                        onClick={clearAllFilters}
                        className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[10px] font-bold border border-red-500/20 text-red-400 hover:bg-red-500/10 transition-all"
                      >
                        <span className="material-symbols-outlined !text-sm">close</span>
                        清除
                      </button>
                    )}

                    {/* Download XLSX button */}
                    <button
                      onClick={handleDownloadXlsx}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[10px] font-bold border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/10 transition-all"
                    >
                      <span className="material-symbols-outlined !text-sm">download</span>
                      下載 XLSX
                    </button>
                  </div>
                </div>

                {/* Table */}
                <div className="overflow-x-auto rounded-xl border border-white/5">
                  <table className="w-full text-[10px] text-left whitespace-nowrap">
                    <thead className="bg-white/5 text-white/40 uppercase sticky top-0 z-10">
                      {/* Column headers with sort */}
                      <tr>
                        <th className="px-3 py-2.5 font-bold">#</th>
                        {displayCols.map(col => {
                          const mid = getModelIdFromCol(col);
                          const c = mid !== null && modelColorMap[mid] ? modelColorMap[mid] : null;
                          const isHighlight = isPredCol(col) || isErrCol(col) || col === 'EAC';
                          const label = getColLabel(col);
                          const isSorted = sortConfig.key === col;
                          return (
                            <th
                              key={col}
                              className={`px-3 py-2.5 font-bold cursor-pointer select-none hover:text-white/60 transition-colors ${isHighlight ? (c ? c.text : 'text-primary') : ''}`}
                              onClick={() => handleSort(col)}
                            >
                              <span className="inline-flex items-center gap-0.5">
                                {label}
                                <SortIcon direction={isSorted ? sortConfig.direction : null} />
                              </span>
                            </th>
                          );
                        })}

                      </tr>

                      {/* Filter row */}
                      {showFilters && (
                        <tr className="bg-white/[0.03] border-t border-white/5">
                          <td className="px-2 py-1.5"></td>
                          {displayCols.map(col => {
                            const numeric = isNumericCol(col);
                            const filterVal = filters[col]?.value || '';
                            const filterOp = filters[col]?.operator || (numeric ? '>' : 'contains');
                            return (
                              <td key={col} className="px-1.5 py-1.5">
                                <div className="flex items-center gap-0.5">
                                  {numeric ? (
                                    <>
                                      <select
                                        value={filterOp}
                                        onChange={(e) => updateFilter(col, 'operator', e.target.value)}
                                        className="bg-white/5 border border-white/10 rounded text-[9px] text-white/60 px-1 py-0.5 w-10 focus:border-primary/50 focus:outline-none"
                                      >
                                        <option value=">">&gt;</option>
                                        <option value="<">&lt;</option>
                                        <option value="=">=</option>
                                        <option value=">=">&gt;=</option>
                                        <option value="<=">&lt;=</option>
                                      </select>
                                      <input
                                        type="text"
                                        value={filterVal}
                                        onChange={(e) => updateFilter(col, 'value', e.target.value)}
                                        placeholder="值"
                                        className="bg-white/5 border border-white/10 rounded text-[9px] text-white/70 px-1.5 py-0.5 w-16 placeholder:text-white/15 focus:border-primary/50 focus:outline-none font-mono"
                                      />
                                    </>
                                  ) : (
                                    <input
                                      type="text"
                                      value={filterVal}
                                      onChange={(e) => {
                                        updateFilter(col, 'operator', 'contains');
                                        updateFilter(col, 'value', e.target.value);
                                      }}
                                      placeholder="搜尋..."
                                      className="bg-white/5 border border-white/10 rounded text-[9px] text-white/70 px-1.5 py-0.5 w-full placeholder:text-white/15 focus:border-primary/50 focus:outline-none"
                                    />
                                  )}
                                </div>
                              </td>
                            );
                          })}

                        </tr>
                      )}
                    </thead>
                    <tbody className="divide-y divide-white/5 font-mono text-white/70">
                      {pagedRows.length === 0 ? (
                        <tr>
                          <td colSpan={displayCols.length + 2} className="px-4 py-8 text-center text-white/20 text-xs">
                            {hasActiveFilters ? '沒有符合篩選條件的資料' : '無資料'}
                          </td>
                        </tr>
                      ) : (
                        pagedRows.map((row, idx) => {
                          const globalIdx = page * PAGE_SIZE + idx;
                          const singleErrPct = result.mode === 'single' ? row.error_pct : null;
                          const rowBg = singleErrPct !== null && singleErrPct !== undefined
                            ? (singleErrPct > 15 ? 'bg-red-500/[0.03]' : singleErrPct > 5 ? 'bg-yellow-500/[0.02]' : '')
                            : '';
                          return (
                            <tr key={globalIdx} className={`hover:bg-white/[0.03] transition-colors ${rowBg}`}>
                              <td className="px-3 py-2 text-white/20">{globalIdx + 1}</td>
                              {displayCols.map(col => {
                                const val = row[col];
                                const mid = getModelIdFromCol(col);
                                const c = mid !== null && modelColorMap[mid] ? modelColorMap[mid] : null;

                                let cellClass = 'px-3 py-2';
                                if (isPredCol(col)) cellClass += ` font-bold ${c ? c.text : 'text-primary'}`;
                                else if (col === 'EAC') cellClass += ' text-blue-400';
                                else if (isErrCol(col)) {
                                  return (
                                    <td key={col} className="px-3 py-2">
                                      <ErrorLight pct={val} />
                                    </td>
                                  );
                                }

                                return (
                                  <td key={col} className={cellClass}>
                                    {val === null || val === undefined ? '—' : typeof val === 'number' ? Number(val).toFixed(4) : String(val)}
                                  </td>
                                );
                              })}

                            </tr>
                          );
                        })
                      )}
                    </tbody>
                  </table>
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-between mt-4 text-[10px] text-white/40">
                    <span>顯示 {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, sortedRows.length)} / {sortedRows.length}{hasActiveFilters ? ` (篩選自 ${result.total_rows} 筆)` : ''}</span>
                    <div className="flex gap-1">
                      <button
                        onClick={() => setPage(p => Math.max(0, p - 1))}
                        disabled={page === 0}
                        className="px-3 py-1 rounded border border-white/10 hover:bg-white/5 disabled:opacity-20 transition-all"
                      >
                        上一頁
                      </button>
                      {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                        let pageNum;
                        if (totalPages <= 7) pageNum = i;
                        else if (page < 3) pageNum = i;
                        else if (page > totalPages - 4) pageNum = totalPages - 7 + i;
                        else pageNum = page - 3 + i;
                        return (
                          <button
                            key={pageNum}
                            onClick={() => setPage(pageNum)}
                            className={`px-2.5 py-1 rounded border transition-all ${page === pageNum ? 'border-primary text-primary bg-primary/10' : 'border-white/10 hover:bg-white/5'}`}
                          >
                            {pageNum + 1}
                          </button>
                        );
                      })}
                      <button
                        onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                        disabled={page >= totalPages - 1}
                        className="px-3 py-1 rounded border border-white/10 hover:bg-white/5 disabled:opacity-20 transition-all"
                      >
                        下一頁
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center space-y-4">
                <div className="size-20 rounded-full bg-white/5 mx-auto flex items-center justify-center">
                  <span className="material-symbols-outlined !text-4xl text-white/10">query_stats</span>
                </div>
                <p className="text-sm font-bold text-white/20 tracking-widest uppercase">上傳資料並選擇模型後即可開始預測</p>
                <p className="text-[10px] text-white/10">可選擇多個模型進行比對分析</p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <div className="p-8 border-t border-white/10 bg-background-dark/95 flex justify-end gap-6 backdrop-blur-xl">
        <button onClick={onBack || onNavigateToDashboard} className="text-xs font-bold text-white/30 hover:text-white transition-colors">回模型訓練</button>
        <button onClick={onNavigateToDashboard} className="px-10 py-3 rounded-xl bg-white/5 border border-white/10 text-white font-bold text-xs hover:bg-white/10 hover:border-white/20 transition-all">返回首頁看板</button>
      </div>
    </div>
  );
}