import React, { useEffect, useState } from 'react';
import Navbar from '../components/Navbar';

const SystemIntroduction = () => (
  <div className="w-full h-full min-h-[250px] bg-white/[0.03] rounded-2xl border border-white/10 p-8 flex flex-col justify-center">
    <div className="flex items-center gap-3 mb-4">
      <div className="size-10 rounded-xl bg-primary/20 text-primary flex items-center justify-center">
        <span className="material-symbols-outlined font-bold">wb_sunny</span>
      </div>
      <h3 className="text-2xl font-black text-white tracking-tight">
        日光預(Sunergy Analytics Lab)：太陽能發電預測系統
      </h3>
    </div>

    <div className="space-y-4 text-white/70 leading-relaxed text-lg">
      <p>
        本系統整合了 <span className="text-primary font-bold">大數據分析</span> 與{' '}
        <span className="text-primary font-bold">深度學習技術</span>，
        專為太陽能案場設計。透過監測日照量、溫度及歷史發電數據，我們能精準預測電力產出，
        並透過自動化資料清洗流程，確保預測模型在不同格式下仍能維持其穩定性與準確度。
      </p>

      <div className="grid grid-cols-2 gap-4 mt-6">
        <div className="p-4 bg-white/5 rounded-xl border border-white/5">
          <h4 className="text-white font-bold mb-1 flex items-center gap-2">
            <span className="material-symbols-outlined text-sm">precision_manufacturing</span>
            自動化訓練
          </h4>
          <p className="text-sm opacity-60">一鍵啟動多模型並行訓練，尋找最佳超參數。</p>
        </div>

        <div className="p-4 bg-white/5 rounded-xl border border-white/5">
          <h4 className="text-white font-bold mb-1 flex items-center gap-2">
            <span className="material-symbols-outlined text-sm">show_chart</span>
            高精度預測
          </h4>
          <p className="text-sm opacity-60">採用 LSTM 與 XGBoost 等演算法，將誤差降至最低。</p>
        </div>
      </div>
    </div>
  </div>
);

export default function Dashboard({
  onLogout,
  onNavigateToTrain,
  onNavigateToDashboard,
  onNavigateToSites,
  onOpenCreateSite,
  onNavigateToPredict,
  onNavigateToModelMgmt,
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [allModels, setAllModels] = useState([]);
  const [loadingModels, setLoadingModels] = useState(true);
  const [modelError, setModelError] = useState('');

  const getStatusColor = (status) => {
    switch (status) {
      case '已部署':
        return 'bg-green-500/20 text-green-400 border border-green-500/30';
      case '閒置中':
        return 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30';
      case '測試中':
        return 'bg-blue-500/20 text-blue-400 border border-blue-500/30';
      case '已建立':
        return 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30';
      default:
        return 'bg-white/10 text-white/60';
    }
  };

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoadingModels(true);
        setModelError('');

        const user = JSON.parse(localStorage.getItem("user"));

        if (!user || !user.user_id) {
          throw new Error("找不到登入資訊，請重新登入");
        }

        const userId = user.user_id;

        const res = await fetch(
          `http://127.0.0.1:8000/train/trained-models?user_id=${userId}`
        );

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();

        const mapped = data.map((item) => {
          const trainedDate = item.trained_at
            ? new Date(item.trained_at).toLocaleDateString('zh-TW')
            : '—';

          return {
            id: item.model_id,
            name: `${item.model_type || '-'}_${item.model_id} ${
              item.site_name && item.location
                ? `${item.site_name}[${item.location}]`
                : item.site_name
                  ? `[${item.site_name}]`
                  : '-'
            }`.trim(),
            type: item.model_type || '-',
            date: trainedDate,
            status: '已建立',
            usage: item.usage_count ?? 0,
            acc: item.parameters?.r2 ?? '—',
            fileName: item.file_name || '未知檔案',
            dataId: item.data_id,
            filePath: item.file_path,
            parameters: item.parameters || {},
          };
        });

        setAllModels(mapped);
      } catch (err) {
        console.error('讀取模型失敗:', err);
        setModelError(err.message || '模型資料載入失敗');
      } finally {
        setLoadingModels(false);
      }
    };

    fetchModels();
  }, []);

  const filteredModels = allModels.filter((model) =>
    model.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const topModels = [...allModels]
    .sort((a, b) => (b.usage || 0) - (a.usage || 0))
    .slice(0, 3);

  return (
    <div className="flex min-h-screen w-full flex-col bg-background-dark text-white font-sans">
      <Navbar
        activePage="dashboard"
        onNavigateToDashboard={onNavigateToDashboard}
        onNavigateToTrain={onNavigateToTrain}
        onNavigateToSites={onNavigateToSites}
        onNavigateToPredict={onNavigateToPredict}
        onNavigateToModelMgmt={onNavigateToModelMgmt}
        onLogout={onLogout}
      />

      <main className="flex-1 w-full max-w-7xl mx-auto p-6 sm:p-10">
        <div className="mb-8 flex items-end justify-between border-b border-white/10 pb-4">
          <div>
            <h1 className="text-3xl font-bold">首頁</h1>
            <p className="text-sm text-white/40">我的案場概況</p>
          </div>
          <button
            onClick={onNavigateToTrain}
            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-background-dark text-sm font-bold transition-transform hover:scale-105"
          >
            <span className="material-symbols-outlined !text-lg font-bold">play_arrow</span>
            開始訓練模型
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <div className="lg:col-span-7 flex flex-col gap-16">
            <section>
              <h2 className="text-xl font-bold mb-4">系統願景</h2>
              <SystemIntroduction />
            </section>

            <section className="bg-white/[0.02] rounded-2xl p-6 border border-white/10">
              <div className="flex justify-between mb-6">
                <h2 className="text-xl font-bold">已建立模型</h2>
                <input
                  type="text"
                  placeholder="搜尋模型..."
                  className="bg-white/5 border border-white/10 rounded-lg py-1 px-4 text-xs focus:outline-none focus:border-primary/40 transition-all"
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>

              <div className="space-y-4 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                {loadingModels ? (
                  <p className="text-sm text-white/50">模型載入中...</p>
                ) : modelError ? (
                  <p className="text-sm text-red-400">{modelError}</p>
                ) : filteredModels.length === 0 ? (
                  <p className="text-sm text-white/50">目前沒有已建立模型</p>
                ) : (
                  filteredModels.map((model) => (
                    <div
                      key={model.id}
                      className="flex justify-between border-b border-white/5 pb-4 last:border-0"
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="text-lg font-bold">{model.name}</h3>
                          <span
                            className={`text-[10px] px-2 py-0.5 rounded font-medium ${getStatusColor(model.status)}`}
                          >
                            {model.status}
                          </span>
                        </div>

                        <p className="text-xs text-white/30 mt-1 font-mono">
                          訓練日期: {model.date}
                        </p>
                        <p className="text-xs text-white/20 mt-1">
                          類型: {model.type} | 使用資料：{model.fileName}
                        </p>
                      </div>

                      <span className="material-symbols-outlined text-white/20 cursor-pointer hover:text-white transition-colors">
                        more_vert
                      </span>
                    </div>
                  ))
                )}
              </div>
            </section>
          </div>

          <div className="lg:col-span-5 flex flex-col gap-8">
            <div className="flex flex-col gap-4 rounded-xl border border-white/10 bg-white/[0.03] p-6 shadow-lg">
              <h3 className="text-base font-medium text-white/80">最佳歷史預測準確度</h3>
              <div className="text-center">
                <p className="text-6xl font-bold text-green-400">98.7%</p>
                <p className="text-sm text-white/60 font-medium">準確度 (MAPE)</p>
              </div>
            </div>

            <div className="flex flex-col gap-4 rounded-xl border border-white/10 bg-gradient-to-br from-green-900/20 to-white/[0.03] p-6 shadow-lg">
              <h3 className="text-base font-medium text-white/80 flex items-center gap-2">
                <span className="material-symbols-outlined text-green-400">eco</span>
                環境減碳效益 (本週)
              </h3>
              <p className="text-3xl font-bold text-green-400">
                2,450 <span className="text-lg font-normal text-white/60">kgCO₂e</span>
              </p>
            </div>

            <section>
              <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                <span className="material-symbols-outlined text-primary">leaderboard</span>
                最常用模型排名
              </h2>

              <div className="flex flex-col gap-4">
                {topModels.length === 0 ? (
                  <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-5">
                    <p className="text-white/40 text-sm">目前尚無模型資料</p>
                  </div>
                ) : (
                  topModels.map((model, index) => (
                    <div
                      key={model.id}
                      className="bg-white/[0.03] border border-white/10 rounded-2xl p-5 relative overflow-hidden group hover:border-primary/50 transition-all"
                    >
                      <div className="flex items-center gap-5">
                        <div
                          className={`text-2xl font-black italic ${
                            index === 0 ? 'text-primary' : 'text-white/20'
                          }`}
                        >
                          0{index + 1}
                        </div>

                        <div className="flex-1">
                          <div className="flex justify-between items-start mb-1">
                            <h3 className="font-bold text-white text-lg">{model.name}</h3>
                            <div className="text-right">
                              <span className="text-[9px] text-white/40 block uppercase leading-none mb-1">
                                使用次數
                              </span>
                              <span className="text-primary font-mono font-bold text-base">
                                {model.usage} 次
                              </span>
                            </div>
                          </div>

                          <div className="flex justify-center my-3">
                            <div className="w-4/5 h-[1px] bg-white/5"></div>
                          </div>

                          <div className="flex justify-between items-end">
                            <div>
                              <p className="text-[10px] text-white/30 uppercase leading-none mb-1">
                                歷史準確度
                              </p>
                              <p className="text-sm font-bold text-green-400">{model.acc}</p>
                            </div>
                            <div>
                              <p className="text-[10px] text-white/30 uppercase leading-none mb-1">
                                模型類型
                              </p>
                              <p className="text-sm text-white/70">{model.type}</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
}