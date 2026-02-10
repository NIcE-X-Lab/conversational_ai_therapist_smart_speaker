import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Mic, Volume2, ShieldCheck, Power } from 'lucide-react';
import './App.css';
import Login from './components/Login';
import SessionControls from './components/SessionControls';

const API_URL = `http://${window.location.hostname}:8000/api`;

function App() {
  const [status, setStatus] = useState({ status: "loading", subject_id: "..." });
  const [turns, setTurns] = useState([]);
  const [isHandsFree, setIsHandsFree] = useState(true);
  const [user, setUser] = useState(null); // { user_id: string }
  const scrollRef = useRef(null);

  const isLoggedIn = !!user;

  // Poll status and turns
  useEffect(() => {
    if (!isLoggedIn) return; // Only poll if logged in

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
  }, [isLoggedIn]); // Depend on login state

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [turns]);

  return (
    <div className="app-container">
      {!isLoggedIn ? (
        <Login
          onLogin={(userData) => setUser(userData)}
          apiUrl={API_URL}
        />
      ) : (
        <>
          {/* Header / Status Card */}
          <div className="status-card">
            <div className="status-row">
              <ShieldCheck className="icon-shield" size={24} />
              <div className="status-text">
                <h2>Session Active</h2>
                <p>User • {user.user_id}</p>
                <p>Session • {status.session_id || "..."}</p>
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
                onChange={async () => {
                  const newMode = !isHandsFree;
                  setIsHandsFree(newMode);
                  try {
                    await axios.post(`${API_URL}/action`, {
                      type: "set_mode",
                      mode: newMode ? "hands_free" : "manual"
                    });
                  } catch (e) { console.error(e); }
                }}
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
            <div
              className={`mic-circle ${status.status === 'listening' ? 'pulse' : ''}`}
              onClick={async () => {
                if (!isHandsFree) {
                  try {
                    await axios.post(`${API_URL}/action`, { type: "start_listening" });
                  } catch (e) { console.error(e); }
                }
              }}
              style={{ cursor: isHandsFree ? 'default' : 'pointer', opacity: isHandsFree ? 0.7 : 1.0 }}
            >
              <Mic size={40} color="white" />
            </div>

            {status.status === 'listening' && (
              <div className="listening-badge">
                Listening...
              </div>
            )}

            <SessionControls
              apiUrl={API_URL}
              onEndSession={() => {
                setUser(null); // Log out
                setTurns([]); // Clear chat
              }}
            />
          </div>
        </>
      )}
    </div>
  );
}

export default App;
