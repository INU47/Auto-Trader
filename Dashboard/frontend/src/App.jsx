import React, { useState, useEffect, useRef } from 'react';
import { createChart, CrosshairMode, CandlestickSeries } from 'lightweight-charts';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import {
  Activity, TrendingUp, TrendingDown, Users, Cpu, MessageSquare, List, BarChart2, Radio, AlertCircle
} from 'lucide-react';

const App = () => {
  const [currentUser, setCurrentUser] = useState('master_trader');
  const [currentSymbol, setCurrentSymbol] = useState('EURUSD');
  const [currentTimeframe, setCurrentTimeframe] = useState('M1');
  const [price, setPrice] = useState(0);
  const [signal, setSignal] = useState('WAITING...');
  const [analysis, setAnalysis] = useState('Waiting for AI signal...');
  const [wsStatus, setWsStatus] = useState('disconnected');
  const [equityData, setEquityData] = useState([]);
  const [detections, setDetections] = useState([]);
  const [activeUsers, setActiveUsers] = useState(['master_trader']);

  const chartContainerRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const chartRef = useRef(null);
  const allDataRef = useRef({});

  // Initialize Lightweight Charts
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        backgroundColor: '#161b22',
        textColor: '#c9d1d9',
      },
      grid: {
        vertLines: { color: '#21262d' },
        horzLines: { color: '#21262d' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: '#30363d',
      },
      timeScale: {
        borderColor: '#30363d',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current.clientWidth });
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // Update chart when symbol/timeframe changes
  useEffect(() => {
    if (!candleSeriesRef.current) return;
    const key = `${currentSymbol}_${currentTimeframe}`;
    const data = allDataRef.current[key] || [];
    candleSeriesRef.current.setData(data);
  }, [currentSymbol, currentTimeframe]);

  // WebSocket Connection
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname === 'localhost' ? 'localhost:8000' : window.location.host;
    const ws = new WebSocket(`${protocol}//${host}/ws`);

    ws.onopen = () => setWsStatus('connected');
    ws.onclose = () => setWsStatus('disconnected');

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);

      if (msg.type === 'history') {
        const historyData = msg.data;
        const newAllData = { ...allDataRef.current };
        const newEquity = [];

        historyData.forEach(item => {
          if (item.type === 'candle') {
            const key = `${item.symbol}_${item.timeframe}`;
            if (!newAllData[key]) newAllData[key] = [];
            newAllData[key].push({
              time: item.time,
              open: parseFloat(item.open),
              high: parseFloat(item.high),
              low: parseFloat(item.low),
              close: parseFloat(item.close),
            });
          }
          // Assuming history might include some equity snapshots or closed trades
          if (item.type === 'trade_closed') {
            newEquity.push({ time: item.time, profit: item.profit });
          }
        });

        // Sort and deduplicate
        Object.keys(newAllData).forEach(key => {
          newAllData[key].sort((a, b) => a.time - b.time);
          newAllData[key] = newAllData[key].filter((v, i, a) => i === 0 || v.time !== a[i - 1].time);
        });

        allDataRef.current = newAllData;
        setEquityData(prev => [...prev, ...newEquity]);

        // Trigger chart update
        const key = `${currentSymbol}_${currentTimeframe}`;
        if (candleSeriesRef.current && newAllData[key]) {
          candleSeriesRef.current.setData(newAllData[key]);
        }
      }

      if (msg.type === 'candle' || msg.type === 'tick') {
        const item = msg;
        const key = `${item.symbol}_${item.timeframe}`;
        const candle = {
          time: item.time,
          open: parseFloat(item.open),
          high: parseFloat(item.high),
          low: parseFloat(item.low),
          close: parseFloat(item.close),
        };

        if (!allDataRef.current[key]) allDataRef.current[key] = [];

        // Update local ref
        const existingIdx = allDataRef.current[key].findIndex(c => c.time === candle.time);
        if (existingIdx !== -1) {
          allDataRef.current[key][existingIdx] = candle;
        } else {
          allDataRef.current[key].push(candle);
        }

        // Live update chart if active
        if (item.symbol === currentSymbol && item.timeframe === currentTimeframe && candleSeriesRef.current) {
          candleSeriesRef.current.update(candle);
          setPrice(candle.close);
        }
      }

      if (msg.type === 'signal') {
        setSignal(msg.action);
        if (msg.analysis) setAnalysis(msg.analysis);

        // Add marker
        if (candleSeriesRef.current) {
          const markers = candleSeriesRef.current.getMarkers() || [];
          markers.push({
            time: msg.time,
            position: msg.action === 'BUY' ? 'belowBar' : 'aboveBar',
            color: msg.action === 'BUY' ? '#26a69a' : '#ef5350',
            shape: msg.action === 'BUY' ? 'arrowUp' : 'arrowDown',
            text: `AI: ${msg.action} (${(msg.confidence * 100).toFixed(0)}%)`,
          });
          candleSeriesRef.current.setMarkers(markers);
        }
      }

      if (msg.type === 'detection') {
        setDetections(prev => [msg, ...prev].slice(0, 10));
      }
    };

    return () => ws.close();
  }, [currentSymbol, currentTimeframe]);

  return (
    <div className="flex flex-col h-screen bg-github-bg text-github-text">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 bg-github-panel border-b border-github-border shadow-soft">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-github-accent/10 rounded-lg">
            <Cpu className="text-github-accent w-6 h-6" />
          </div>
          <h1 className="text-xl font-bold bg-gradient-to-r from-github-accent to-purple-400 bg-clip-text text-transparent">
            QUANT AI VISION
          </h1>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 px-3 py-1 bg-github-bg border border-github-border rounded-full text-xs font-medium">
            <div className={`w-2 h-2 rounded-full shadow-sm ${wsStatus === 'connected' ? 'bg-github-success animate-pulse' : 'bg-github-danger'}`}></div>
            {wsStatus.toUpperCase()}
          </div>

          <div className="flex gap-2">
            <div className="flex items-center gap-2 bg-github-bg border border-github-border rounded-md px-3 py-1 group focus-within:border-github-accent transition-all">
              <Users className="w-3 h-3 text-github-text/40 group-focus-within:text-github-accent" />
              <select
                value={currentUser}
                onChange={(e) => setCurrentUser(e.target.value)}
                className="bg-transparent text-sm outline-none cursor-pointer"
              >
                {activeUsers.map(u => <option key={u} value={u}>{u}</option>)}
              </select>
            </div>

            <select
              value={currentSymbol}
              onChange={(e) => setCurrentSymbol(e.target.value)}
              className="bg-github-bg border border-github-border rounded-md px-3 py-1.5 text-sm outline-none focus:border-github-accent transition-all"
            >
              <option value="EURUSD">EURUSD</option>
              <option value="GBPUSD">GBPUSD</option>
              <option value="USDJPY">USDJPY</option>
              <option value="XAUUSD">XAUUSD</option>
            </select>

            <select
              value={currentTimeframe}
              onChange={(e) => setCurrentTimeframe(e.target.value)}
              className="bg-github-bg border border-github-border rounded-md px-3 py-1.5 text-sm outline-none focus:border-github-accent transition-all"
            >
              <option value="M1">M1 (1 Min)</option>
              <option value="M5">M5 (5 Min)</option>
              <option value="H1">H1 (1 Hour)</option>
            </select>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex overflow-hidden p-4 gap-4">
        {/* Left Column: Charts */}
        <div className="flex-1 flex flex-col gap-4 overflow-y-auto">
          {/* Main Price Chart */}
          <div className="bg-github-panel border border-github-border rounded-xl p-1 relative min-h-[500px]">
            <div className="absolute top-4 left-4 z-10 flex items-center gap-2 bg-github-panel/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-github-border shadow-lg">
              <span className="text-sm font-semibold">{currentSymbol}</span>
              <span className="text-xs text-github-text/60">({currentTimeframe})</span>
            </div>
            <div ref={chartContainerRef} className="w-full h-[500px] rounded-lg overflow-hidden"></div>
          </div>

          {/* Performance Charts Area */}
          <div className="grid grid-cols-2 gap-4 h-64">
            <div className="bg-github-panel border border-github-border rounded-xl p-4 flex flex-col">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp className="text-github-success w-4 h-4" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-github-text/50">Performance Overview</h3>
              </div>
              <div className="flex-1 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={equityData}>
                    <defs>
                      <linearGradient id="colorProfit" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#58a6ff" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#58a6ff" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#21262d" vertical={false} />
                    <XAxis dataKey="time" hide />
                    <YAxis hide domain={['auto', 'auto']} />
                    <RechartsTooltip
                      contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', fontSize: '12px' }}
                      itemStyle={{ color: '#58a6ff' }}
                    />
                    <Area type="monotone" dataKey="profit" stroke="#58a6ff" fillOpacity={1} fill="url(#colorProfit)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-github-panel border border-github-border rounded-xl p-4">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Activity className="text-github-accent w-4 h-4" />
                  <h3 className="text-xs font-bold uppercase tracking-wider text-github-text/50">Market Pulse</h3>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <span className="text-[10px] text-github-text/40 uppercase">Last Price</span>
                  <div className="text-2xl font-mono font-bold">{price.toFixed(5)}</div>
                </div>
                <div className="space-y-1 text-right">
                  <span className="text-[10px] text-github-text/40 uppercase">AI Signal</span>
                  <div className={`text-xl font-bold ${signal === 'BUY' ? 'text-github-success' : signal === 'SELL' ? 'text-github-danger' : 'text-github-text'}`}>
                    {signal}
                  </div>
                </div>
              </div>
              <div className="mt-6 pt-6 border-t border-github-border/50">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-github-text/40 italic">Active Strategy: CONSERVATIVE</span>
                  <span className="px-2 py-0.5 bg-github-success/10 text-github-success rounded-full text-[10px]">VERIFIED</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: AI Analysis & Logs */}
        <div className="w-80 flex flex-col gap-4">
          {/* Intelligence Panel */}
          <div className="flex-1 bg-github-panel border border-github-border rounded-xl flex flex-col overflow-hidden">
            <div className="p-4 border-b border-github-border bg-github-bg/30">
              <div className="flex items-center gap-2">
                <MessageSquare className="text-github-accent w-4 h-4" />
                <h3 className="text-xs font-bold uppercase tracking-wider">AI Intelligence Panel</h3>
              </div>
            </div>
            <div className="flex-1 p-4 overflow-y-auto custom-scrollbar">
              <div className="p-4 bg-github-bg/50 rounded-lg border border-github-border/50 mb-4 transition-all hover:border-github-accent/50 group">
                <div className="flex items-center gap-2 text-[10px] text-github-text/40 mb-2 font-mono">
                  <Radio className="w-3 h-3 text-github-accent animate-pulse" />
                  LIVE_FEED / {currentSymbol}
                </div>
                <p className="text-sm leading-relaxed text-github-text/90 italic">
                  "{analysis}"
                </p>
                <div className="mt-3 text-[10px] text-github-accent font-bold group-hover:underline cursor-pointer">
                  VIEW FULL SUMMARY →
                </div>
              </div>

              {/* Detections List */}
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-xs font-bold text-github-text/30">
                  <List className="w-3 h-3" />
                  REAL-TIME DETECTIONS
                </div>
                {detections.map((det, i) => (
                  <div key={i} className="flex flex-col gap-1 p-3 bg-github-panel rounded-lg border border-github-border/30 hover:bg-github-bg/50 transition-colors">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-bold text-github-accent">{det.label}</span>
                      <span className="text-[9px] font-mono text-github-text/40">
                        {new Date(det.time * 1000).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="text-[10px] text-github-text/60">
                      CONFIDENCE: {(det.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
                {detections.length === 0 && (
                  <div className="text-center py-10 text-github-text/20 italic text-xs">
                    No detections yet...
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Quick Actions / System Health */}
          <div className="bg-github-panel border border-github-border rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <BarChart2 className="text-github-accent w-4 h-4" />
              <h3 className="text-xs font-bold uppercase tracking-wider">System Health</h3>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center text-xs">
                <span className="text-github-text/40">RL Brain Status</span>
                <span className="text-github-success font-bold font-mono">OPTIMAL</span>
              </div>
              <div className="w-full h-1.5 bg-github-bg rounded-full overflow-hidden">
                <div className="h-full bg-github-success w-[85%]"></div>
              </div>
              <div className="flex justify-between items-center text-[10px] text-github-text/50">
                <span>Memory usage: 420MB</span>
                <span>Latency: ~45ms</span>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer Info */}
      <footer className="px-6 py-2 bg-github-bg border-t border-github-border flex justify-between items-center text-[10px] text-github-text/30">
        <div>&copy; 2026 HYBRID QUANT SYSTEM | BUILD v94.2.react</div>
        <div className="flex gap-4 uppercase font-mono tracking-tighter">
          <span>MT5_CONNECTED: YES</span>
          <span>POSTGRES: OK</span>
          <span>LLM_QUOTA: 100%</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
