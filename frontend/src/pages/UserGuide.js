// src/pages/UserGuide.js
import React, { useState, useEffect } from 'react';

export default function UserGuide({ onFinish }) {
  const [step, setStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  // 定義教學步驟資料

  const steps = [
    {
      id: 'start',
      icon: 'school',
      title: '歡迎來到日光預',
      desc: '點擊畫面任一處，開始您的綠能預測之旅。',
      screenContent: (
        <div className="flex h-full flex-col items-center justify-center bg-background-dark text-white">
           <span className="text-4xl font-black text-primary tracking-widest mb-2">日光預</span>
           <p className="text-sm text-white/50">AI Solar Prediction Platform</p>
        </div>
      )
    },
    {
      id: 1,
      icon: 'login',
      title: 'Step 1: 登入帳號',
      desc: '註冊並登入平台，為您的太陽能案場建立專屬的數位雙生模型。',
      screenContent: (
        <div className="flex h-full flex-col items-center justify-center bg-[#1E1E1E]">
           <div className="w-3/4 space-y-3">
              <div className="h-2 w-1/3 bg-white/20 rounded"></div>
              <div className="h-8 w-full bg-black/30 rounded border border-white/10"></div>
              <div className="h-8 w-full bg-black/30 rounded border border-white/10"></div>
              <div className="h-8 w-full bg-primary rounded shadow-lg flex items-center justify-center text-background-dark font-bold text-xs">LOGIN</div>
           </div>
        </div>
      )
    },
    {
      id: 2,
      icon: 'cloud_upload',
      title: 'Step 2: 上傳案場資料',
      desc: '資料太雜亂？交給我們處理。輕鬆上傳歷史數據，系統自動完成清理與校正。',
      screenContent: (
        <div className="flex h-full flex-col items-center justify-center bg-[#1E1E1E] relative overflow-hidden">
           <div className="absolute inset-0 flex items-center justify-center">
              <div className="size-32 border-2 border-dashed border-white/20 rounded-xl flex items-center justify-center">
                 <span className="material-symbols-outlined text-4xl text-white/20">upload_file</span>
              </div>
           </div>
           <div className="absolute animate-bounce-once top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
              <span className="material-symbols-outlined text-5xl text-primary drop-shadow-[0_0_15px_rgba(242,204,13,0.8)]">description</span>
           </div>
        </div>
      )
    },
    {
      id: 3,
      icon: 'model_training', 
      title: 'Step 3: 進行模型訓練', // 已更新
      desc: '選擇最適合的演算法（如 XGBoost、SVR、RandomForest），讓機器學習自動尋找數據特徵並完成模型開發。',
      screenContent: (
        <div className="flex h-full flex-col items-center justify-center bg-[#1E1E1E]">
           <div className="flex gap-4 mb-4">
              <div className="size-12 rounded-lg border border-primary bg-primary/20 flex items-center justify-center text-primary"><span className="material-symbols-outlined">neurology</span></div>
              <div className="size-12 rounded-lg border border-white/10 bg-white/5 flex items-center justify-center text-white/30"><span className="material-symbols-outlined">forest</span></div>
           </div>
           <div className="w-3/4 h-1.5 bg-white/10 rounded-full overflow-hidden">
              <div className="h-full bg-primary w-2/3 animate-[width_1s_ease-in-out_infinite]"></div>
           </div>
           <p className="text-[10px] text-primary mt-2 font-mono">Model Training...</p>
        </div>
      )
    },
    {
      id: 4,
      icon: 'insights',
      title: 'Step 4: 利用訓練好的模型進行預測', // 已更新
      desc: '一鍵導入預測數據。使用您專屬的模型精準推算未來發電量，並生成專業的發電量結果。',
      screenContent: (
        <div className="flex h-full flex-col items-center justify-center bg-[#1E1E1E] p-4">
           <div className="w-full flex justify-between items-end h-20 gap-1 mb-2">
              <div className="w-1/5 h-[40%] bg-white/10 rounded-t"></div>
              <div className="w-1/5 h-[60%] bg-white/10 rounded-t"></div>
              <div className="w-1/5 h-[80%] bg-primary/50 rounded-t"></div>
              <div className="w-1/5 h-[100%] bg-primary rounded-t shadow-[0_0_15px_#f2cc0d]"></div>
              <div className="w-1/5 h-[50%] bg-white/10 rounded-t"></div>
           </div>
           <div className="flex gap-2 w-full">
              <div className="h-2 w-1/2 bg-white/20 rounded"></div>
              <div className="h-2 w-1/2 bg-white/20 rounded"></div>
           </div>
        </div>
      )
    },
    {
      id: 5,
      icon: 'history',
      title: 'Step 5: 管理歷史記錄',
      desc: '凡走過必留下痕跡。完整的訓練歷程都會被保存，隨時比較成效，持續優化。',
      screenContent: (
        <div className="flex h-full flex-col items-center justify-center bg-[#1E1E1E] p-4 space-y-2">
           <div className="w-full h-8 rounded border border-white/10 flex items-center px-2 gap-2">
              <div className="size-4 rounded-full bg-primary"></div>
              <div className="h-1 w-20 bg-white/20 rounded"></div>
           </div>
           <div className="w-full h-8 rounded border border-white/10 flex items-center px-2 gap-2 opacity-60">
              <div className="size-4 rounded-full bg-white/20"></div>
              <div className="h-1 w-20 bg-white/20 rounded"></div>
           </div>
           <div className="w-full h-8 rounded border border-white/10 flex items-center px-2 gap-2 opacity-30">
              <div className="size-4 rounded-full bg-white/20"></div>
              <div className="h-1 w-20 bg-white/20 rounded"></div>
           </div>
        </div>
      )
    },
    {
      id: 'end',
      icon: 'rocket_launch',
      title: '準備好了嗎？',
      desc: '立即開始使用日光預，最大化您的太陽能收益。',
      screenContent: (
        <div className="flex h-full flex-col items-center justify-center bg-background-dark text-white">
           <div className="size-16 rounded-full bg-primary text-background-dark flex items-center justify-center animate-bounce">
              <span className="material-symbols-outlined text-4xl">check</span>
           </div>
        </div>
      )
    }
  ];

  const handleNext = () => {
    if (isAnimating) return;
    
    if (step < steps.length - 1) {
      setIsAnimating(true);
      setTimeout(() => setIsAnimating(false), 500); // 防止連點
      setStep(step + 1);
    } else {
      onFinish(); // 結束教學，回首頁
    }
  };

  // 鍵盤支援
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'Enter') {
        handleNext();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [step]);

  const currentStepData = steps[step];

  return (
    <div 
      onClick={handleNext}
      className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-background-dark text-white cursor-pointer overflow-hidden select-none"
    >
      {/* 背景裝飾光暈 */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 size-[500px] bg-primary/5 rounded-full blur-[100px] pointer-events-none"></div>

      {/* 進度條 */}
      <div className="absolute top-8 w-64 h-1 bg-white/10 rounded-full overflow-hidden">
        <div 
          className="h-full bg-primary transition-all duration-500 ease-out"
          style={{ width: `${((step + 1) / steps.length) * 100}%` }}
        ></div>
      </div>

      {/* 主內容區 */}
      <div className="relative z-10 flex flex-col items-center gap-12 max-w-4xl w-full px-4">
        
        {/* 筆電 Mockup */}
        <div className="relative w-full max-w-lg aspect-[16/10] transition-transform duration-700 ease-out hover:scale-[1.02]">
           {/* 螢幕外框 */}
           <div className="absolute inset-0 bg-[#2a2a2a] rounded-t-2xl border-4 border-[#444] border-b-0 shadow-2xl overflow-hidden">
              {/* 螢幕內容 (帶切換動畫) */}
              <div key={step} className="w-full h-full animate-fade-in">
                 {currentStepData.screenContent}
              </div>
           </div>
           {/* 筆電底座 */}
           <div className="absolute -bottom-4 left-[-5%] w-[110%] h-4 bg-[#333] rounded-b-xl shadow-[0_10px_30px_rgba(0,0,0,0.5)]">
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-16 h-1.5 bg-[#222] rounded-b-md"></div>
           </div>
        </div>

        {/* 文字說明區 (帶浮現動畫) */}
        <div key={step} className="text-center flex flex-col items-center gap-4 animate-slide-up">
           <div className="flex items-center justify-center size-16 rounded-full bg-white/5 border border-white/10 mb-2 text-primary">
              <span className="material-symbols-outlined !text-3xl">{currentStepData.icon}</span>
           </div>
           <h2 className="text-3xl md:text-4xl font-bold text-white">
             {currentStepData.title}
           </h2>
           <p className="text-lg text-white/60 max-w-xl leading-relaxed">
             {currentStepData.desc}
           </p>
        </div>

      </div>

      {/* 底部提示 */}
      <div className="absolute bottom-10 text-white/30 text-sm animate-pulse">
        {step === steps.length - 1 ? '點擊畫面結束教學' : '點擊畫面繼續'}
      </div>

      {/* 跳過按鈕 */}
      <button 
        onClick={(e) => { e.stopPropagation(); onFinish(); }}
        className="absolute top-8 right-8 text-white/40 hover:text-white transition-colors text-sm font-medium px-4 py-2"
      >
        跳過教學
      </button>

    </div>
  );
}