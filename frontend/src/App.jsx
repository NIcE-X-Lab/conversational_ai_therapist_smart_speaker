import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Mic, Volume2, RotateCcw, ShieldCheck, Power } from 'lucide-react';
import './App.css';

const API_URL = "http://localhost:8000/api";

function App() {
  const [status, setStatus] = useState({ status: "loading", subject_id: "..." });
  const [turns, setTurns] = useState([]);
  const [isHandsFree, setIsHandsFree] = useState(true);
  const scrollRef = useRef(null);

  // Poll status and turns
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const statusRes = await axios.get(`${API_URL}/status`);
        setStatus(statusRes.data);

        const turnsRes = await axios.get(`${API_URL}/turns`);
        // Only update if length changed to avoid jitter, or just update always?
        // React handles diffing.
        setTurns(turnsRes.data);
      } catch (error) {
        console.error("API Error:", error);
      }
    }, 1000); // Poll every 1s
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [turns]);

  return (
    <div className="app-container">
      {/* Header / Status Card */}
      <div className="status-card">
        <div className="status-row">
          <ShieldCheck className="icon-shield" size={24} />
          <div className="status-text">
            <h2>Session ready</h2>
            <p>Subject • {status.subject_id}</p>
          </div>
          <div className="sound-badge">
            <Volume2 size={16} /> Sound on
          </div>
        </div>
      </div>

      {/* Mode Toggle */}
      <div className="mode-toggle">
        <span>Interactive Hands-Free Mode</span>
        <label className="switch">
          <input
            type="checkbox"
            checked={isHandsFree}
            onChange={() => setIsHandsFree(!isHandsFree)}
          />
          <span className="slider round"></span>
        </label>
      </div>

      {/* Chat Area */}
      <div className="chat-area">
        {turns.map((turn, index) => (
          <div
            key={index}
            className={`message-bubble ${turn.speaker === 'agent' ? 'agent' : 'user'}`}
          >
            {turn.text}
          </div>
        ))}
        {/* Dummy listening indicator if needed */}
        <div ref={scrollRef} />
      </div>

      {/* Bottom Controls / Mic */}
      <div className="bottom-controls">
        <div className="mic-circle pulse">
          <Mic size={40} color="white" />
        </div>
        <div className="assistant-status">
          <h3>Assistant is speaking...</h3>
          <p>We will start listening automatically after the assistant finishes speaking.</p>
        </div>
        <div className="control-bar">
          <div className="control-item">
            <div className="listening-badge">
              <span className="wave">|||</span> Listening...
            </div>
          </div>
          <div className="control-item stop-btn">
            <Power size={20} /> Stop
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
