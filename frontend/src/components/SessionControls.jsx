import { useState } from 'react';
import axios from 'axios';
import { Play, Pause, XCircle } from 'lucide-react';

function SessionControls({ apiUrl, onEndSession }) {
    const [paused, setPaused] = useState(false);

    const togglePause = async () => {
        try {
            const endpoint = paused ? '/resume' : '/pause';
            await axios.post(`${apiUrl}${endpoint}`);
            setPaused(!paused);
        } catch (e) {
            console.error("Pause/Resume failed:", e);
        }
    };

    const handleEndSession = async () => {
        if (!window.confirm("Are you sure you want to end the session?")) return;

        try {
            await axios.post(`${apiUrl}/end_session`);
            onEndSession(); // Reset frontend state
        } catch (e) {
            console.error("End Session failed:", e);
        }
    };

    return (
        <div className="session-controls">
            <button
                className={`control-btn ${paused ? 'resume' : 'pause'}`}
                onClick={togglePause}
            >
                {paused ? <Play size={20} /> : <Pause size={20} />}
                {paused ? " Resume" : " Pause"}
            </button>

            <button
                className="control-btn end-session"
                onClick={handleEndSession}
            >
                <XCircle size={20} /> End Session
            </button>
        </div>
    );
}

export default SessionControls;
