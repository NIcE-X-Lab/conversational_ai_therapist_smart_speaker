import { useState } from 'react';
import axios from 'axios';
import { User, UserPlus } from 'lucide-react';
import '../App.css'; // Reuse styles or create new CSS

function Login({ onLogin, apiUrl }) {
    const [loading, setLoading] = useState(false);

    const handleLogin = async (userId) => {
        setLoading(true);
        try {
            const res = await axios.post(`${apiUrl}/login`, { user_id: userId });
            if (res.data.status === "logged_in") {
                onLogin(res.data);
            }
        } catch (err) {
            console.error("Login failed:", err);
            alert("Login failed. Check backend connection.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="login-container">
            <h1>AI Therapist Login</h1>
            <p>Select a user to begin session</p>

            <div className="login-buttons">
                <button
                    className="login-btn test-user"
                    onClick={() => handleLogin('test_user')}
                    disabled={loading}
                >
                    <User size={24} /> Test User
                </button>

                <button
                    className="login-btn new-user"
                    onClick={() => handleLogin('new_user')}
                    disabled={loading}
                >
                    <UserPlus size={24} /> New User
                </button>
            </div>
            {loading && <p>Connecting...</p>}
        </div>
    );
}

export default Login;
