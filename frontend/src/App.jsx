
import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [isLogin, setIsLogin] = useState(true);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
  });

  const [question, setQuestion] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);

  // Authentication form
  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  // Login and registration
  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage("");
    setLoading(true);

    const endpoint = isLogin ? "/auth/login" : "/auth/register";

    try {
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(
          isLogin
            ? {
                email: formData.email,
                password: formData.password,
              }
            : formData
        ),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Something went wrong");
      }

      if (isLogin) {
        setUser(data.user);
        setChatHistory([]);
      } else {
        setMessage("Registration successful! Please log in.");
        setIsLogin(true);
        setFormData({
          name: "",
          email: formData.email,
          password: "",
        });
      }
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  };

  // Send question to FastAPI chatbot
  const handleSendMessage = async (e) => {
    e.preventDefault();

    const userQuestion = question.trim();

    if (!userQuestion || chatLoading) return;

    setQuestion("");
    setChatLoading(true);

    setChatHistory((prev) => [
      ...prev,
      {
        role: "user",
        content: userQuestion,
      },
    ]);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: userQuestion,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to process your question");
      }

      setChatHistory((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
        },
      ]);
    } catch (error) {
      setChatHistory((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${error.message}`,
          sources: [],
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  // Logout
  const handleLogout = () => {
    setUser(null);
    setChatHistory([]);
    setQuestion("");
    setMessage("");
    setFormData({
      name: "",
      email: "",
      password: "",
    });
    setIsLogin(true);
  };

  // =====================================
  // CHATBOT PAGE - AFTER LOGIN
  // =====================================

  if (user) {
    return (
      <div className="chat-app">

        <nav className="chat-navbar">
          <div className="chat-brand">
            <div className="chat-logo">T</div>
            <h2>TweetSense <span>AI</span></h2>
          </div>

          <div className="navbar-user">
            <span>Welcome, {user.name}</span>
            <button onClick={handleLogout}>Logout</button>
          </div>
        </nav>

        <main className="chat-main">

          <div className="chat-heading">
            <div className="ai-badge">✦ AI POWERED PRODUCT INSIGHTS</div>
            <h1>Explore Product Opinions</h1>
            <p>
              Ask questions and discover what people really think
              about products through tweets.
            </p>
          </div>

          <section className="chat-container">

            <div className="chat-messages">

              {chatHistory.length === 0 && (
                <div className="empty-chat">
                  <div className="empty-icon">✦</div>
                  <h2>Welcome to TweetSense AI!</h2>
                  <p>
                    Ask me about laptop brands and product opinions
                    based on real tweets.
                  </p>

                  <div className="suggestion-list">
                    {[
                      "Is Dell laptop good?",
                      "What are the advantages of HP laptops?",
                      "What do people think about ASUS laptops?",
                    ].map((item) => (
                      <button
                        key={item}
                        onClick={() => setQuestion(item)}
                      >
                        {item} →
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {chatHistory.map((chat, index) => (
                <div
                  className={`message-row ${chat.role}`}
                  key={index}
                >
                  <div className="message-content">
                    <div className="message-label">
                      {chat.role === "user" ? "You" : "TweetSense AI"}
                    </div>

                    <div className="message-text">
                      {chat.content}
                    </div>

                    {chat.sources?.length > 0 && (
                      <div className="sources-section">
                        <h3>Supporting Tweets</h3>

                        {chat.sources.map((tweet, i) => (
                          <div className="source-card" key={i}>
                            <div className="source-top">
                              <strong>Tweet {i + 1}</strong>
                              <span>
                                Similarity: {(tweet.similarity * 100).toFixed(1)}%
                              </span>
                            </div>

                            <p>{tweet.text}</p>

                            <small>Brand: {tweet.brand}</small>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {chatLoading && (
                <div className="message-row assistant">
                  <div className="message-content">
                    <div className="message-label">TweetSense AI</div>
                    <div className="typing-indicator">
                      Analyzing tweets and generating response...
                    </div>
                  </div>
                </div>
              )}

            </div>

            <form className="chat-input-area" onSubmit={handleSendMessage}>
              <input
                type="text"
                placeholder="Ask about a laptop or product..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                disabled={chatLoading}
                maxLength={500}
              />

              <button
                type="submit"
                disabled={!question.trim() || chatLoading}
              >
                {chatLoading ? "..." : "Send ↑"}
              </button>
            </form>

            <p className="chat-disclaimer">
              AI-generated insights are based on retrieved tweets
              and may not represent every customer's experience.
            </p>

          </section>
        </main>
      </div>
    );
  }

  // =====================================
  // LOGIN AND REGISTRATION PAGE
  // =====================================

  return (
    <div className="auth-container">

      <div className="auth-left">
        <div className="brand">
          <div className="brand-icon">T</div>
          <h2>TweetSense <span>AI</span></h2>
        </div>

        <div className="hero-content">
          <div className="hero-tag">AI POWERED PRODUCT INSIGHTS</div>

          <h1>
            Understand what
            <br />
            people really think.
          </h1>

          <p>
            Discover real product opinions through tweets.
            Ask questions and get intelligent AI-powered insights
            backed by social media conversations.
          </p>

          <div className="feature-list">
            <div>✓ Real tweet-based opinions</div>
            <div>✓ Intelligent AI analysis</div>
            <div>✓ Relevant supporting tweets</div>
          </div>
        </div>

        <div className="left-footer">
          © 2026 TweetSense AI. All rights reserved.
        </div>
      </div>

      <div className="auth-right">
        <div className="auth-box">

          <div className="mobile-brand">
            <h2>TweetSense <span>AI</span></h2>
          </div>

          <div className="auth-heading">
            <h1>{isLogin ? "Welcome back!" : "Create an account"}</h1>
            <p>
              {isLogin
                ? "Sign in to continue to your account."
                : "Join TweetSense AI and explore product opinions."}
            </p>
          </div>

          <div className="auth-tabs">
            <button
              className={isLogin ? "active" : ""}
              onClick={() => {
                setIsLogin(true);
                setMessage("");
              }}
            >
              Login
            </button>

            <button
              className={!isLogin ? "active" : ""}
              onClick={() => {
                setIsLogin(false);
                setMessage("");
              }}
            >
              Register
            </button>
          </div>

          <form onSubmit={handleSubmit}>

            {!isLogin && (
              <div className="form-group">
                <label>Full Name</label>
                <input
                  type="text"
                  name="name"
                  placeholder="Enter your full name"
                  value={formData.name}
                  onChange={handleChange}
                  minLength={2}
                  maxLength={100}
                  required
                />
              </div>
            )}

            <div className="form-group">
              <label>Email Address</label>
              <input
                type="email"
                name="email"
                placeholder="you@example.com"
                value={formData.email}
                onChange={handleChange}
                required
              />
            </div>

            <div className="form-group">
              <label>Password</label>
              <input
                type="password"
                name="password"
                placeholder="Enter your password"
                value={formData.password}
                onChange={handleChange}
                minLength={isLogin ? 1 : 8}
                required
              />
            </div>

            {message && (
              <div className="message">{message}</div>
            )}

            <button
              type="submit"
              className="primary-btn"
              disabled={loading}
            >
              {loading
                ? "Please wait..."
                : isLogin
                  ? "Sign In"
                  : "Create Account"}
            </button>

          </form>

          <div className="auth-switch">
            {isLogin
              ? "Don't have an account? "
              : "Already have an account? "}

            <button
              onClick={() => {
                setIsLogin(!isLogin);
                setMessage("");
              }}
            >
              {isLogin ? "Register" : "Login"}
            </button>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;
