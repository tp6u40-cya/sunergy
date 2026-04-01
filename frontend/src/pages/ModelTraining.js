// src/pages/ModelTraining.js
import React, { useEffect, useState } from 'react';
import Navbar from '../components/Navbar';

// --- 輔助組件：警示燈 ---
const StatusLight = ({ wmape }) => {
  const value = parseFloat(wmape);
  let status = { color: 'bg-green-500', label: '正常', shadow: 'shadow-[0_0_10px_rgba(34,197,94,0.8)]' };
  if (value > 0.15) status = { color: 'bg-red-500', label: '異常', shadow: 'shadow-[0_0_10px_rgba(239,68,68,0.8)]' };
  else if (value > 0.05) status = { color: 'bg-yellow-500', label: '需留意', shadow: 'shadow-[0_0_10px_rgba(234,179,8,0.8)]' };

  return (
    <div className="flex items-center gap-2">
      <div className={`size-3 rounded-full ${status.color} ${status.shadow}`}></div>
      <span className="text-xs text-white/80">{status.label}</span>
    </div>
  );
};


const IntervalSlider = ({ label, min, max, start, end, onStartChange, onEndChange, step = 1 }) => {
  const startPct = ((start - min) / (max - min)) * 100;
  const endPct = ((end - min) / (max - min)) * 100;
  const handleStartMove = (v) => onStartChange(Math.min(v, end));
  const handleEndMove = (v) => onEndChange(Math.max(v, start));

  return (
    <div className="mb-8">
      <div className="flex justify-between items-center mb-3">
        <label className="text-xs font-bold text-primary">{label}</label>
      </div>
      <div className="relative h-6 w-full flex items-center mb-4">
        <div className="absolute h-1 w-full bg-white/10 rounded-full"></div>
        <div className="absolute h-1 bg-primary z-10" style={{ left: `${startPct}%`, right: `${100 - endPct}%` }}></div>
        <input type="range" min={min} max={max} step={step} value={end} onChange={(e) => handleEndMove(Number(e.target.value))} className="absolute w-full h-full appearance-none bg-transparent pointer-events-none z-30 accent-white [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:size-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-primary" />
        <input type="range" min={min} max={max} step={step} value={start} onChange={(e) => handleStartMove(Number(e.target.value))} className="absolute w-full h-full appearance-none bg-transparent pointer-events-none z-20 accent-white [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:size-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-primary" />
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-[9px] text-white/40 mb-1">起始設定</p>
          <input type="number" step={step} value={start} onChange={(e) => onStartChange(Number(e.target.value))} className="w-full bg-white/5 border border-white/10 rounded p-2 text-xs text-center focus:border-primary outline-none" />
        </div>
        <div>
          <p className="text-[9px] text-white/40 mb-1">結束設定 (MAX: {max})</p>
          <input type="number" step={step} value={end} onChange={(e) => onEndChange(Number(e.target.value))} className="w-full bg-white/5 border border-white/10 rounded p-2 text-xs text-center focus:border-primary outline-none" />
        </div>
      </div>
    </div>
  );
};

export default function ModelTraining({ onBack, onNext, onNavigateToDashboard, onNavigateToPredict, onLogout, onNavigateToSites }) {
  const [splitRatio, setSplitRatio] = useState(80);
  const [selectedModels, setSelectedModels] = useState(['XGBoost']);
  const [paramIntervals, setParamIntervals] = useState({
    // XGBoost
    XGB_trees_s: 100, XGB_trees_e: 500,
    XGB_depth_s: 3, XGB_depth_e: 10,
    XGB_lr_s: 0.01, XGB_lr_e: 0.3,
    // XGBoost extra params (grid/manual)
    XGB_subsample_s: 0.5, XGB_subsample_e: 1.0,
    XGB_colsample_s: 0.5, XGB_colsample_e: 1.0,
    XGB_min_child_s: 0, XGB_min_child_e: 6,
    XGB_lambda_s: 0.0, XGB_lambda_e: 2.0,
    XGB_alpha_s: 0.0, XGB_alpha_e: 1.0,
    // SVR
    SVR_c_s: 1, SVR_c_e: 50,
    SVR_epsilon_s: 0.01, SVR_epsilon_e: 1.0,
    SVR_gamma: 'scale',
    // RandomForest
    RF_trees_s: 50, RF_trees_e: 300,
    RF_depth_s: 3, RF_depth_e: 12,
  });
  // Cap for XGB grid combinations
  const [xgbMaxComb, setXgbMaxComb] = useState(100);
  const [activeChartLines, setActiveChartLines] = useState({ XGBoost: true, SVR: true, RandomForest: true });
  const [isTraining, setIsTraining] = useState(false);
  const [isTrained, setIsTrained] = useState(false);
  const [trainingResults, setTrainingResults] = useState({});
  const [strategy, setStrategy] = useState('bayes');
  const [device, setDevice] = useState('auto');  // 'auto' | 'cpu' | 'cuda'
  const [cleanedFileName, setCleanedFileName] = useState('');
  const [bayesTrials, setBayesTrials] = useState(30);
  // Training progress state
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState('');


  const toggleModel = (id) => setSelectedModels(prev => prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]);

  // 載入目前清洗檔案資訊
  React.useEffect(() => {
    const dataId =
      localStorage.getItem('afterDataId') ||
      localStorage.getItem('lastDataId');

    if (!dataId) return;

    fetch(`http://127.0.0.1:8000/train/info?data_id=${dataId}`)
      .then(res => res.json())
      .then(data => {
        if (data.file_name) {
          setCleanedFileName(data.file_name);
        }
      })
      .catch(err => console.error(err));
  }, []);

  const handleStartTraining = async () => {
    if (selectedModels.length === 0) return alert('請選擇模型');
    const dataId =
      localStorage.getItem('afterDataId') ||
      localStorage.getItem('lastDataId');
    if (!dataId) return alert('找不到清洗後的資料來源');

    const params = {};
    if (selectedModels.includes('XGBoost')) params['XGBoost'] = {
      n_estimators: { start: paramIntervals.XGB_trees_s, end: paramIntervals.XGB_trees_e, step: 100 },
      max_depth: { start: paramIntervals.XGB_depth_s, end: paramIntervals.XGB_depth_e, step: 1 },
      learning_rate: { start: Number(paramIntervals.XGB_lr_s), end: Number(paramIntervals.XGB_lr_e), step: 0.01 },
      subsample: { start: Number(paramIntervals.XGB_subsample_s), end: Number(paramIntervals.XGB_subsample_e), step: 0.05 },
      colsample_bytree: { start: Number(paramIntervals.XGB_colsample_s), end: Number(paramIntervals.XGB_colsample_e), step: 0.05 },
      min_child_weight: { start: Number(paramIntervals.XGB_min_child_s), end: Number(paramIntervals.XGB_min_child_e), step: 1 },
      reg_lambda: { start: Number(paramIntervals.XGB_lambda_s), end: Number(paramIntervals.XGB_lambda_e), step: 0.1 },
      reg_alpha: { start: Number(paramIntervals.XGB_alpha_s), end: Number(paramIntervals.XGB_alpha_e), step: 0.1 },
      _max_combinations: Number(xgbMaxComb),
    };
    if (selectedModels.includes('RandomForest')) params['RandomForest'] = {
      n_estimators: { start: paramIntervals.RF_trees_s, end: paramIntervals.RF_trees_e, step: 50 },
      max_depth: { start: paramIntervals.RF_depth_s, end: paramIntervals.RF_depth_e, step: 1 },
    };
    if (selectedModels.includes('SVR')) {
      const cStep = Math.max(1, Math.floor((paramIntervals.SVR_c_e - paramIntervals.SVR_c_s) / 5) || 1);
      params['SVR'] = {
        C: { start: paramIntervals.SVR_c_s, end: paramIntervals.SVR_c_e, step: cStep },
        epsilon: { start: Number(paramIntervals.SVR_epsilon_s), end: Number(paramIntervals.SVR_epsilon_e), step: 0.05 },
        gamma: { values: ['scale', 'auto'] },
      };
    }


    // Add _trials parameter for Bayesian optimization strategy
    if (strategy === 'bayes') {
      Object.keys(params).forEach(modelKey => {
        params[modelKey]._trials = Number(bayesTrials);
      });
    }

    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingStatus('準備訓練資料...');

    // Simulate progress updates since backend doesn't support streaming
    const progressInterval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev < 90) {
          const increment = Math.random() * 10 + 2;
          return Math.min(prev + increment, 90);
        }
        return prev;
      });
    }, 500);

    // Update status based on strategy
    setTimeout(() => setTrainingStatus(`執行 ${strategy === 'bayes' ? 'Bayesian 優化' : strategy === 'grid' ? '網格搜索' : '手動參數'} 訓練中...`), 1000);

    try {
      const res = await fetch('http://127.0.0.1:8000/train/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data_id: Number(dataId), split_ratio: Number(splitRatio) / 100, models: selectedModels, strategy, params, device })
      });

      clearInterval(progressInterval);
      setTrainingProgress(95);
      setTrainingStatus('處理訓練結果...');

      const json = await res.json();
      console.log('Training API response:', json); // Debug log
      if (!res.ok) throw new Error(json?.detail || '訓練失敗');

      // Check if results exist and have data
      const results = json.results || {};
      console.log('Training results:', results); // Debug log

      setTrainingProgress(100);
      setTrainingStatus('訓練完成！');

      if (Object.keys(results).length === 0) {
        alert('訓練完成但沒有返回結果，請檢查後端日誌');
      } else {
        setTrainingResults(results);
        setIsTrained(true);
      }
      if (json.cleaned_file) setCleanedFileName(json.cleaned_file);
    } catch (e) {
      clearInterval(progressInterval);
      setTrainingProgress(0);
      setTrainingStatus('');
      console.error('Training error:', e); // Debug log
      alert(e.message || '訓練失敗');
    } finally {
      setIsTraining(false);
    }
  };
  const metricExplanations = [
  {
    id: 'R² Score',
    name: '決定係數',
    desc: '衡量模型解釋資料變異的能力。越接近 1 代表模型擬合度越高，預測越準確。',
    formula: '1 - (SS_res / SS_tot)',
    color: 'text-green-400'
  },
  {
    id: 'RMSE',
    name: '均方根誤差',
    desc: '衡量預測值與真實值間的平均差距。單位與發電量一致 (kW)，數值越小代表誤差越低。',
    formula: 'sqrt(mean((y_true - y_pred)²))',
    color: 'text-blue-400'
  },
  {
    id: 'MAE',
    name: '平均絕對殘差',
    desc: '預測值與真實值差距的絕對值平均。較不受極端值影響，能直觀反映平均誤差大小。',
    formula: 'mean(abs(y_true - y_pred))',
    color: 'text-purple-400'
  },
  {
    id: 'WMAPE',
    name: '加權平均絕對百分比誤差',
    desc: '將總絕對誤差除以總實際發電量。相對於 MAPE，它解決了實際值為 0 時的計算問題。',
    formula: 'sum(abs(error)) / sum(abs(actual))',
    color: 'text-yellow-500'
  }
];
  return (
    <div className="min-h-screen w-full bg-background-dark text-white flex flex-col font-sans">
      <Navbar activePage="predict" onNavigateToDashboard={onNavigateToDashboard} onNavigateToPredict={onNavigateToPredict} onLogout={onLogout} onNavigateToSites={onNavigateToSites} />

      {/* [新增] Sticky Header 步驟指示器 */}
      <div className="w-full border-b border-white/10 bg-white/[.02] px-6 py-3 sticky top-[64px] sm:top-[65px] z-40 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <button onClick={onBack} className="flex items-center gap-1 text-sm text-white/50 hover:text-white transition-colors">
            <span className="material-symbols-outlined !text-lg">arrow_back</span>
            返回上一步
          </button>

          <div className="text-sm font-medium">
            <span className="text-white/40">1. 上傳資料</span>
            <span className="mx-2 text-white/30">/</span>
            <span className="text-white/40">2. 清理資料</span>
            <span className="mx-2 text-white/30">/</span>
            <span className="text-white/40 ">3. 調整單位</span>
            <span className="mx-2 text-white/30">/</span>
            <span className="text-primary font-bold">4. 模型訓練</span>

          </div>
        </div>
      </div>

      <main className="flex-1 w-full max-w-7xl mx-auto p-6 py-10 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* 左側：設定流程 */}
        <div className="lg:col-span-4 flex flex-col gap-8">
          {/* Step 1: 分割資料 */}
          <section className="bg-white/[0.02] p-5 rounded-2xl border border-white/5">
            <h2 className="text-sm font-bold text-primary mb-4 flex items-center gap-2">
              <span className="size-5 rounded-full bg-primary text-background-dark flex items-center justify-center text-[10px]">1</span> 分割資料
            </h2>
            <div className="flex justify-between text-[10px] text-white/50 mb-2"><span>訓練集 {splitRatio}%</span><span>測試集 {100 - splitRatio}%</span></div>
            <input type="range" min="50" max="95" step="5" value={splitRatio} onChange={(e) => setSplitRatio(e.target.value)} className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary" />
          </section>

          {/* Info: 目前處理檔案 */}
          <section className="bg-white/[0.02] p-5 rounded-2xl border border-white/5">
            <h2 className="text-sm font-bold text-primary mb-2">目前處理檔案</h2>
            <p className="text-xs text-white/70 break-all">{cleanedFileName || "-"}</p>
          </section>
          {/* Step 2: 選擇模型 */}
          <section className="bg-white/[0.02] p-5 rounded-2xl border border-white/5">
            <h2 className="text-sm font-bold text-primary mb-4 flex items-center gap-2">
              <span className="size-5 rounded-full bg-primary text-background-dark flex items-center justify-center text-[10px]">2</span> 選擇模型
            </h2>
            <div className="grid grid-cols-2 gap-2">
              {['XGBoost', 'SVR', 'RandomForest'].map(m => (
                <button key={m} onClick={() => toggleModel(m)} className={`p-2 rounded-lg border text-[10px] font-bold transition-all ${selectedModels.includes(m) ? 'border-primary bg-primary/10 text-primary' : 'border-white/10 bg-white/5 text-white/40'}`}>{m}</button>
              ))}
            </div>
          </section>

          {/* Step 3: 調整參數 */}
          <section className="bg-white/[0.02] p-5 rounded-2xl border border-white/5">
            <h2 className="text-sm font-bold text-primary mb-4 flex items-center gap-2">
              <span className="size-5 rounded-full bg-primary text-background-dark flex items-center justify-center text-[10px]">3</span> 調整參數
            </h2>
            {/* 策略選擇 */}
            <div className="mb-4 flex items-center gap-2 text-[10px]">
              <span className="text-white/50">策略選擇</span>
              {["grid", "bayes"].map((sg) => (
                <label key={sg} className={["px-2", "py-1", "rounded", "border", "cursor-pointer", (strategy === sg ? "border-primary text-primary" : "border-white/10 text-white/50")].join(' ')}>
                  <input type="radio" name="strategy" value={sg} checked={strategy === sg} onChange={() => setStrategy(sg)} className="hidden" />
                  {sg.toUpperCase()}
                </label>
              ))}
              {strategy === 'bayes' && (
                <div className="flex items-center gap-2 ml-4">
                  <span className="text-white/50">TRIALS</span>
                  <input type="number" min={5} max={200} value={bayesTrials} onChange={(e) => setBayesTrials(Number(e.target.value))} className="w-20 bg-white/5 border border-white/10 rounded p-1 text-xs text-center focus:border-primary outline-none" />
                </div>
              )}
            </div>

            <div className="flex flex-col gap-4">
              {selectedModels.map(id => (
                <div key={id} className="p-4 bg-black/20 rounded-xl border border-white/5">
                  <p className="text-[10px] font-bold text-white/60 mb-6 uppercase tracking-tighter border-b border-white/5 pb-1">{id} 模型參數設定</p>

                  {id === 'XGBoost' && (
                    <>
                      <IntervalSlider label="n_estimators" min={10} max={2000} start={paramIntervals.XGB_trees_s} end={paramIntervals.XGB_trees_e} onStartChange={(v) => setParamIntervals({ ...paramIntervals, XGB_trees_s: v })} onEndChange={(v) => setParamIntervals({ ...paramIntervals, XGB_trees_e: v })} />
                      <IntervalSlider label="max_depth" min={1} max={16} start={paramIntervals.XGB_depth_s} end={paramIntervals.XGB_depth_e} onStartChange={(v) => setParamIntervals({ ...paramIntervals, XGB_depth_s: v })} onEndChange={(v) => setParamIntervals({ ...paramIntervals, XGB_depth_e: v })} />
                      <IntervalSlider step={0.01} label="learning_rate" min={0.01} max={0.3} start={paramIntervals.XGB_lr_s} end={paramIntervals.XGB_lr_e} onStartChange={(v) => setParamIntervals({ ...paramIntervals, XGB_lr_s: v })} onEndChange={(v) => setParamIntervals({ ...paramIntervals, XGB_lr_e: v })} />
                      <IntervalSlider step={0.05} label="subsample" min={0.5} max={1.0} start={paramIntervals.XGB_subsample_s} end={paramIntervals.XGB_subsample_e} onStartChange={(v) => setParamIntervals({ ...paramIntervals, XGB_subsample_s: v })} onEndChange={(v) => setParamIntervals({ ...paramIntervals, XGB_subsample_e: v })} />
                      <IntervalSlider step={0.05} label="colsample_bytree" min={0.5} max={1.0} start={paramIntervals.XGB_colsample_s} end={paramIntervals.XGB_colsample_e} onStartChange={(v) => setParamIntervals({ ...paramIntervals, XGB_colsample_s: v })} onEndChange={(v) => setParamIntervals({ ...paramIntervals, XGB_colsample_e: v })} />
                      <IntervalSlider step={1} label="min_child_weight" min={0} max={10} start={paramIntervals.XGB_min_child_s} end={paramIntervals.XGB_min_child_e} onStartChange={(v) => setParamIntervals({ ...paramIntervals, XGB_min_child_s: v })} onEndChange={(v) => setParamIntervals({ ...paramIntervals, XGB_min_child_e: v })} />
                      <IntervalSlider step={0.1} label="reg_lambda" min={0.0} max={5.0} start={paramIntervals.XGB_lambda_s} end={paramIntervals.XGB_lambda_e} onStartChange={(v) => setParamIntervals({ ...paramIntervals, XGB_lambda_s: v })} onEndChange={(v) => setParamIntervals({ ...paramIntervals, XGB_lambda_e: v })} />
                      <IntervalSlider step={0.1} label="reg_alpha" min={0.0} max={5.0} start={paramIntervals.XGB_alpha_s} end={paramIntervals.XGB_alpha_e} onStartChange={(v) => setParamIntervals({ ...paramIntervals, XGB_alpha_s: v })} onEndChange={(v) => setParamIntervals({ ...paramIntervals, XGB_alpha_e: v })} />
                      {strategy === 'grid' && (function () {
                        const cnt = (s, e, st) => {
                          const step = Math.max(Number(st) || 0, 0.000001);
                          return Math.max(1, Math.floor(((Number(e) - Number(s)) / step) + 0.000001) + 1);
                        };
                        const combos =
                          cnt(paramIntervals.XGB_trees_s, paramIntervals.XGB_trees_e, 100) *
                          cnt(paramIntervals.XGB_depth_s, paramIntervals.XGB_depth_e, 1) *
                          cnt(paramIntervals.XGB_lr_s, paramIntervals.XGB_lr_e, 0.01) *
                          cnt(paramIntervals.XGB_subsample_s, paramIntervals.XGB_subsample_e, 0.05) *
                          cnt(paramIntervals.XGB_colsample_s, paramIntervals.XGB_colsample_e, 0.05) *
                          cnt(paramIntervals.XGB_min_child_s, paramIntervals.XGB_min_child_e, 1) *
                          cnt(paramIntervals.XGB_lambda_s, paramIntervals.XGB_lambda_e, 0.1) *
                          cnt(paramIntervals.XGB_alpha_s, paramIntervals.XGB_alpha_e, 0.1);
                        const over = combos > xgbMaxComb;
                        return (
                          <div className="mt-2 text-[10px] text-white/60 flex items-center gap-3">
                            <div className="flex items-center gap-1">
                              <span className="text-white/40">組合上限</span>
                              <input type="number" min={10} max={1000} step={10} value={xgbMaxComb} onChange={(e) => setXgbMaxComb(Number(e.target.value) || 10)} className="w-20 bg-white/5 border border-white/10 rounded p-1 text-[10px] text-center focus:border-primary outline-none" />
                            </div>
                            <div className={over ? "text-red-400" : "text-white/50"}>預估組合數：{combos}{over ? `（將抽樣至 ${xgbMaxComb} 組）` : ''}</div>
                          </div>
                        );
                      })()}
                    </>
                  )}
                  {id === 'SVR' && (
                    <>
                      <IntervalSlider label="C (懲罰參數)" min={0.1} max={100} step={0.1} start={paramIntervals.SVR_c_s} end={paramIntervals.SVR_c_e} onStartChange={(v) => setParamIntervals({ ...paramIntervals, SVR_c_s: v })} onEndChange={(v) => setParamIntervals({ ...paramIntervals, SVR_c_e: v })} />
                      <IntervalSlider label="epsilon (容忍度)" min={0.001} max={2.0} step={0.01} start={paramIntervals.SVR_epsilon_s} end={paramIntervals.SVR_epsilon_e} onStartChange={(v) => setParamIntervals({ ...paramIntervals, SVR_epsilon_s: v })} onEndChange={(v) => setParamIntervals({ ...paramIntervals, SVR_epsilon_e: v })} />
                      <div className="mb-6">
                        <label className="text-xs font-bold text-primary mb-2 block">gamma (核函數係數)</label>
                        <div className="flex gap-2">
                          {['scale', 'auto'].map(g => (
                            <button key={g} onClick={() => setParamIntervals({ ...paramIntervals, SVR_gamma: g })} className={`px-3 py-1.5 rounded border text-[10px] font-bold transition-all ${paramIntervals.SVR_gamma === g ? 'border-primary bg-primary/10 text-primary' : 'border-white/10 bg-white/5 text-white/40'}`}>{g}</button>
                          ))}
                        </div>
                      </div>
                    </>
                  )}
                  {id === 'RandomForest' && (
                    <>
                      <IntervalSlider label="n_estimators (森林規模)" min={10} max={1000} start={paramIntervals.RF_trees_s} end={paramIntervals.RF_trees_e} onStartChange={(v) => setParamIntervals({ ...paramIntervals, RF_trees_s: v })} onEndChange={(v) => setParamIntervals({ ...paramIntervals, RF_trees_e: v })} />
                      <IntervalSlider label="max_depth (最大深度)" min={1} max={30} step={1} start={paramIntervals.RF_depth_s} end={paramIntervals.RF_depth_e} onStartChange={(v) => setParamIntervals({ ...paramIntervals, RF_depth_s: v })} onEndChange={(v) => setParamIntervals({ ...paramIntervals, RF_depth_e: v })} />
                    </>
                  )}
                </div>
              ))}
              {selectedModels.length === 0 && <p className="text-center text-[10px] text-white/20 italic py-4">請先在步驟 2 選擇模型</p>}
            </div>
          </section>

          <button onClick={handleStartTraining} disabled={isTraining || selectedModels.length === 0} className="w-full py-4 bg-primary text-background-dark rounded-xl font-black text-sm transition-all hover:scale-[1.02] active:scale-95 disabled:opacity-30">
            {isTraining ? '模型訓練中...' : '開始執行訓練'}
          </button>

          {/* Training Progress Bar */}
          {isTraining && (
            <div className="mt-4 p-4 bg-white/[0.02] rounded-xl border border-white/5">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-white/60">{trainingStatus}</span>
                <span className="text-xs font-bold text-primary">{Math.round(trainingProgress)}%</span>
              </div>
              <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-primary to-green-400 rounded-full transition-all duration-300 ease-out"
                  style={{ width: `${trainingProgress}%` }}
                />
              </div>
              <div className="mt-2 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                <span className="text-[10px] text-white/40">正在處理 {selectedModels.join(', ')} ...</span>
              </div>
            </div>
          )}
        </div>

        {/* 右側：結果顯示 */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          <div className={`flex-1 rounded-2xl border border-white/10 bg-white/[0.01] p-6 relative overflow-hidden ${!isTrained && 'flex items-center justify-center border-dashed'}`}>
            {isTrained ? (
              <div className="w-full animate-fade-in">
                <div className="flex justify-between items-center mb-8 border-b border-white/5 pb-4">
                  <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                    <span className="material-symbols-outlined text-primary">assessment</span>
                    訓練結果看板
                  </h2>
                </div>

                {/* <SolarProductionChart results={trainingResults} activeLines={activeChartLines} /> */}

                <div className="mt-4 overflow-hidden rounded-xl border border-white/10 shadow-2xl bg-black/20">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-white/5 text-white/60 uppercase tracking-wider">
                      <tr>
                        <th className="px-6 py-4 font-bold">模型</th>
                        <th className="px-6 py-4 font-bold">狀態</th>
                        <th className="px-6 py-4 font-bold text-right">R² Score</th>
                        <th className="px-6 py-4 font-bold text-right">RMSE (kW)</th>
                        <th className="px-6 py-4 font-bold text-right">MAE (kW)</th>
                        <th className="px-6 py-4 font-bold text-right">WMAPE</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5 font-mono text-white/90">
                      {Object.values(trainingResults).map(res => (
                        res.status === 'error' ? (
                          <tr key={res.id} className="hover:bg-white/5 transition-colors bg-red-500/5">
                            <td className="px-6 py-5 font-bold text-red-400 font-sans">{res.id}</td>
                            <td colSpan={5} className="px-6 py-5 text-red-400 text-base">
                              <div className="flex items-center gap-2">
                                <span className="material-symbols-outlined !text-lg">error</span>
                                {res.error || '訓練失敗'}
                              </div>
                            </td>
                          </tr>
                        ) : (
                          <tr key={res.id} className="hover:bg-white/5 transition-colors">
                            <td className="px-6 py-5 font-bold text-primary font-sans text-lg">{res.id}</td>
                            <td className="px-6 py-5"><StatusLight wmape={res.wmape} /></td>
                            <td className="px-6 py-5 text-right text-green-400 text-base font-bold">{res.r2 !== undefined ? Number(res.r2).toFixed(3) : '-'}</td>
                            <td className="px-6 py-5 text-right text-base">{res.rmse !== undefined ? Number(res.rmse).toFixed(3) : '-'}</td>
                            <td className="px-6 py-5 text-right text-base">{res.mae !== undefined ? Number(res.mae).toFixed(3) : '-'}</td>
                            <td className="px-6 py-5 text-right text-yellow-500 text-base font-bold">{res.wmape !== undefined ? Number(res.wmape).toFixed(4) : '-'}</td>
                          </tr>
                        )
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Best Params 展示 */}
                <div className="mt-10 flex flex-col gap-6">
  {Object.values(trainingResults)
    .filter(r => r.status === 'ok' && r.best_params && Object.keys(r.best_params).length > 0)
    .map(res => (
      <div 
        key={res.id} 
        className="w-full p-8 rounded-2xl border border-white/10 bg-white/[0.03] shadow-lg animate-fade-in"
      >
        {/* 標題部分 */}
        <div className="flex items-center justify-between mb-6 border-b border-primary/20 pb-4">
          <h4 className="text-lg font-bold text-primary flex items-center gap-3">
            <span className="material-symbols-outlined !text-2xl">tune</span>
            {res.id} 最佳參數設定結果
          </h4>
          {/* <span className="text-[10px] text-white/30 uppercase tracking-[0.2em]">Optimized Parameters</span> */}
        </div>

        {/* 參數網格：增加欄數 (md:grid-cols-3 或 4) 讓橫向空間更充裕 */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-x-10 gap-y-4">
          {Object.entries(res.best_params)
            .filter(([k]) => !k.startsWith('_'))
            .map(([key, val]) => (
              <div key={key} className="flex justify-between items-center py-2 border-b border-white/5">
                <span className="text-sm text-white/40">{key}</span>
                <span className="text-sm text-white font-mono font-black">
                  {typeof val === 'number' 
                    ? (Number.isInteger(val) ? val : val.toFixed(4)) 
                    : String(val)}
                </span>
              </div>
            ))}
        </div>
      </div>
    ))}
</div>
<div className="mt-12 pt-8 border-t border-white/10">
  <div className="flex items-center gap-2 mb-6 text-white/60">
    <span className="material-symbols-outlined !text-xl">help_outline</span>
    <h3 className="text-sm font-bold tracking-widest uppercase">模型評估指標說明</h3>
  </div>
  
{/* --- 模型評估指標說明 --- */}
<div className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-8">
  {metricExplanations.map((metric) => (
    <div 
      key={metric.id} 
      className="p-8 rounded-2xl border border-white/10 bg-white/[0.04] hover:border-primary/40 transition-all group"
    >
      <div className="flex flex-col gap-3 mb-5">
        <div className="flex items-center gap-4">
          {/* ID 設為最大：text-2xl (約 24px) */}
          <span className={`text-2xl font-black px-4 py-1 rounded-xl ${metric.bg} ${metric.color} tracking-tight`}>
            {metric.id}
          </span>
          {/* Name 設為次大：text-lg (約 18px) */}
          <h5 className="text-lg font-bold text-white/90">
            {metric.name}
          </h5>
        </div>
      </div>
      
      {/* 描述文字：text-lg (約 18px) */}
      <p className="text-lg text-white/60 leading-relaxed">
        {metric.desc}
      </p>
    </div>
  ))}
</div>
</div>
              </div>
            ) : (
              <div className="text-center opacity-20 py-20">
                <span className="material-symbols-outlined !text-8xl mb-4">query_stats</span>
                <h1 className="text-2xl font-bold text-white mb-2">等待模型訓練</h1>
                <p className="text-white/60 text-base max-w-sm mx-auto leading-relaxed">請在左側面板配置參數並選擇模型，點擊「開始執行訓練」以查看詳細分析報告。</p>
              </div>
            )}
          </div>
        </div>
      </main>

      <div className="p-6 border-t border-white/10 bg-background-dark/90 flex justify-end gap-4">
        <button onClick={onBack} className="px-6 py-2 text-white/40 hover:text-white transition-colors text-sm">取消</button>
        <button onClick={onNext} disabled={!isTrained} className={`px-10 py-2 rounded-lg font-bold text-sm transition-all ${isTrained ? 'bg-primary text-background-dark hover:shadow-[0_0_15px_rgba(242,204,13,0.4)]' : 'bg-white/10 text-white/20 cursor-not-allowed'}`}>開始進行預測</button>
      </div>
    </div>
  );
}