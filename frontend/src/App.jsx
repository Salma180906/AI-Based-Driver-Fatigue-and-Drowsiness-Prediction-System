import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldCheck,
  AlertTriangle,
  AlertOctagon,
  Camera,
  Film,
  Volume2,
  VolumeX,
  Play,
  Square,
  Crosshair,
  BarChart3,
  Activity,
  Eye,
  Clock,
  Compass,
  X
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8765';

// Used only when the backend is unreachable, so the scenario buttons still appear
const FALLBACK_DEMOS = [
  { id: 'nthu_normal_alert', title: 'Normal Alert Driving', description: '', expected_state: 'ALERT', filename: 'nthu_normal_alert.mp4' },
  { id: 'nthu_drowsy_microsleep', title: 'Drowsy Microsleep', description: '', expected_state: 'DROWSINESS', filename: 'nthu_drowsy_microsleep.mp4' },
  { id: 'nthu_yawning_nodding', title: 'Yawning Episode', description: '', expected_state: 'DROWSINESS', filename: 'nthu_yawning_nodding.mp4' },
  { id: 'nthu_distracted_driving', title: 'Distracted Driving', description: '', expected_state: 'DISTRACTION', filename: 'nthu_distracted_driving.mp4' }
];

export default function App() {
  // Navigation & Stream State
  const [streamSource, setStreamSource] = useState('webcam'); // 'webcam' | 'nthu_demo' | 'upload'
  const [selectedDemoVideo, setSelectedDemoVideo] = useState('nthu_normal_alert.mp4');
  const [demoVideos, setDemoVideos] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [backendOk, setBackendOk] = useState(true);
  const [videoError, setVideoError] = useState('');
  const [isMuted, setIsMuted] = useState(false);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [showMetricsModal, setShowMetricsModal] = useState(false);
  const [modelMetrics, setModelMetrics] = useState(null);

  // Multilingual Voice Alert Settings (English & Tamil)
  const [voiceLanguage, setVoiceLanguage] = useState('en'); // 'en' | 'ta'
  const [activeVoiceNotice, setActiveVoiceNotice] = useState(null);
  const [cooldownSec, setCooldownSec] = useState(0);

  // Synchronous refs to prevent stale closure during requestAnimationFrame
  const voiceLanguageRef = useRef(voiceLanguage);
  voiceLanguageRef.current = voiceLanguage;
  const isMutedRef = useRef(isMuted);
  isMutedRef.current = isMuted;

  // Voice Alert Cooldown & Repetition Tracking Refs
  const lastVoiceAlertTimeRef = useRef(0);
  const voiceAlertCountRef = useRef(0);
  const currentVoiceIncidentRef = useRef(null);
  const normalStreakStartRef = useRef(null);
  const activeAudioRef = useRef(null);

  // Calibration state
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibProgress, setCalibProgress] = useState(0);

  // Live Telemetry from Backend
  const [telemetry, setTelemetry] = useState({
    face_detected: false,
    state: 'ALERT',
    fatigue_score: 0.0,
    fatigue_trend: 'Stable',
    buzzer: false,
    buzzer_reason: null,
    is_drowsy: false,
    is_distracted: false,
    distraction_duration: 0.0,
    ear: 0.28,
    EAR_mean: 0.28,
    EAR_min: 0.26,
    mar: 0.25,
    MAR_mean: 0.25,
    pitch: 0.0,
    yaw: 0.0,
    roll: 0.0,
    perclos: 0.04,
    blink_frequency: 16.0,
    avg_blink_duration: 0.20,
    blink_count: 4,
    yawn_count: 0,
    head_nod_count: 0,
    gaze_direction: 'Center / Forward',
    gaze_stability: 0.95,
    baseline_summary: {
      baseline_ear: 0.28,
      dynamic_ear_threshold: 0.22,
      baseline_blink_duration: 0.20
    },
    deviations: {
      EAR_deviation_from_baseline: 0.0,
      blink_duration_ratio: 1.0,
      pitch_deviation: 0.0
    },
    models: {
      'Random Forest': 0.12,
      'SVM (RBF Kernel)': 0.15,
      'Logistic Regression': 0.10,
      'Ensemble Consensus': 0.13
    },
    landmarks_px: []
  });

  // DOM References
  const videoRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const hiddenCanvasRef = useRef(null);
  const audioContextRef = useRef(null);
  const isProcessingRef = useRef(false);
  const animationFrameRef = useRef(null);

  // Initialize Web Audio API
  useEffect(() => {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        audioContextRef.current = new AudioCtx();
      }
    } catch (e) {
      console.warn('Web Audio not initialized:', e);
    }
    return () => {
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Switching language takes effect immediately: stop any clip playing in the old language
  useEffect(() => {
    if (activeAudioRef.current) {
      activeAudioRef.current.pause();
      activeAudioRef.current.currentTime = 0;
      activeAudioRef.current = null;
    }
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    // allow the next detected incident to be announced in the new language right away
    lastVoiceAlertTimeRef.current = 0;
    voiceAlertCountRef.current = 0;
  }, [voiceLanguage]);

  // Fetch available NTHU demo videos & model evaluation metrics
  useEffect(() => {
    fetch(`${API_BASE}/api/demo-videos`)
      .then((res) => res.json())
      .then((data) => {
        setDemoVideos(Array.isArray(data) ? data : []);
        setBackendOk(true);
      })
      .catch((err) => {
        console.log('Could not load demo videos:', err);
        setBackendOk(false);
        setDemoVideos(FALLBACK_DEMOS);
      });

    fetch(`${API_BASE}/api/model-comparison`)
      .then((res) => res.json())
      .then((data) => setModelMetrics(data))
      .catch((err) => console.log('Could not load model metrics:', err));
  }, []);

  // Lock background scroll when benchmark metrics modal is open
  useEffect(() => {
    if (showMetricsModal) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [showMetricsModal]);

  // Authentic Native Audio Voice Alerts (Tamil & English with Web Speech Fallback)
  const speakDriverAlert = (alertType, lang) => {
    if (isMutedRef.current || !alertType) return;

    let text = '';
    let phoneticFallback = '';

    if (lang === 'ta') {
      if (alertType === 'eyes_closed') {
        text = 'கண்களை திறக்கவும்! கண்களை திறக்கவும்!';
        phoneticFallback = 'Kankalai thiravungal! Kankalai thiravungal!';
      } else if (alertType === 'yawning') {
        text = 'கொட்டாவி விடாதீர்கள், ஓய்வெடுங்கள்! கொட்டாவி விடாதீர்கள், ஓய்வெடுங்கள்!';
        phoneticFallback = 'Kottaavi vidaatheergal! Kottaavi vidaatheergal!';
      } else if (alertType === 'head_nodding') {
        text = 'தூங்குகிறீர்கள், ஓய்வெடுங்கள்! தூங்குகிறீர்கள், ஓய்வெடுங்கள்!';
        phoneticFallback = 'Thoongugireergal, oyvedungal! Thoongugireergal!';
      } else if (alertType === 'distraction') {
        text = 'நேராகப் பாருங்கள், கவனமாக ஓட்டுங்கள்! நேராகப் பாருங்கள், கவனமாக ஓட்டுங்கள்!';
        phoneticFallback = 'Neeraaga paarungal, kavanamaaga oottungal!';
      }
    } else {
      if (alertType === 'eyes_closed') {
        text = 'Open your eyes! Open your eyes!';
        phoneticFallback = text;
      } else if (alertType === 'yawning') {
        text = "Don't yawn, take rest! Don't yawn, take rest!";
        phoneticFallback = text;
      } else if (alertType === 'head_nodding') {
        text = 'You are asleep, take rest! You are asleep, take rest!';
        phoneticFallback = text;
      } else if (alertType === 'distraction') {
        text = 'Turn straight, focus on driving! Turn straight, focus on driving!';
        phoneticFallback = text;
      }
    }

    if (!text) return;

    setActiveVoiceNotice({
      type: alertType,
      text: text,
      lang: lang,
      time: Date.now()
    });

    // 1. Primary: Play authentic pre-recorded audio file (guarantees crystal-clear native Tamil speech)
    try {
      if (activeAudioRef.current) {
        activeAudioRef.current.pause();
        activeAudioRef.current.currentTime = 0;
      }

      const audioUrl = `/audio/${lang}_${alertType}.mp3`;
      const fallbackUrl = `${API_BASE}/api/voice-alert-audio/${lang}_${alertType}.mp3`;
      const audio = new Audio(audioUrl);
      activeAudioRef.current = audio;

      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise.catch((err) => {
          console.log('Static audio play deferred, trying API endpoint:', err);
          const fallbackAudio = new Audio(fallbackUrl);
          activeAudioRef.current = fallbackAudio;
          fallbackAudio.play().catch((err2) => {
            console.warn('Backend audio failed, falling back to Web Speech API:', err2);
            playWebSpeechFallback(text, lang, phoneticFallback);
          });
        });
      }
    } catch (e) {
      playWebSpeechFallback(text, lang, phoneticFallback);
    }
  };

  const playWebSpeechFallback = (text, lang, phoneticFallback) => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return;
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance();
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;

      const voices = window.speechSynthesis.getVoices();
      if (lang === 'ta') {
        const tamilVoice = voices.find(
          (v) =>
            v.lang.toLowerCase().includes('ta') ||
            v.name.toLowerCase().includes('tamil') ||
            v.name.toLowerCase().includes('valluvar')
        );
        if (tamilVoice) {
          utterance.voice = tamilVoice;
          utterance.lang = 'ta-IN';
          utterance.text = text;
        } else {
          const indianVoice = voices.find(
            (v) =>
              v.lang.toLowerCase().includes('en-in') ||
              v.name.toLowerCase().includes('india') ||
              v.name.toLowerCase().includes('heera') ||
              v.name.toLowerCase().includes('ravi')
          );
          if (indianVoice) utterance.voice = indianVoice;
          utterance.lang = 'en-IN';
          utterance.text = phoneticFallback;
        }
      } else {
        const enVoice = voices.find(
          (v) =>
            v.lang.startsWith('en') &&
            (v.name.includes('Natural') ||
              v.name.includes('Google') ||
              v.name.includes('Microsoft') ||
              v.name.includes('David') ||
              v.name.includes('Zira'))
        );
        if (enVoice) utterance.voice = enVoice;
        utterance.lang = 'en-US';
        utterance.text = text;
      }

      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn('Speech synthesis alert failed:', e);
    }
  };

  // Start Video Stream based on active source
  const startStream = async () => {
    setVideoError('');
    setIsRunning(true);
    // Reset backend temporal aggregators for a clean test session
    fetch(`${API_BASE}/api/reset-state`, { method: 'POST' }).catch(() => {});

    if (streamSource === 'webcam') {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, frameRate: { ideal: 30 } }
        });
        if (videoRef.current) {
          videoRef.current.crossOrigin = 'anonymous';
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (err) {
        alert('Could not access physical webcam: ' + err.message + '\nSwitch to NTHU Benchmark Videos to test immediately.');
        setIsRunning(false);
      }
    } else if (streamSource === 'nthu_demo') {
      if (videoRef.current) {
        videoRef.current.crossOrigin = 'anonymous';
        videoRef.current.srcObject = null;
        videoRef.current.src = `${API_BASE}/api/demo-video/${selectedDemoVideo}`;
        videoRef.current.play().catch((e) => console.log('Autoplay handled:', e));
      }
    }
  };

  // Stop Video Stream
  const stopStream = () => {
    setIsRunning(false);
    fetch(`${API_BASE}/api/reset-state`, { method: 'POST' }).catch(() => {});
    if (videoRef.current) {
      if (videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((t) => t.stop());
        videoRef.current.srcObject = null;
      }
      videoRef.current.pause();
    }
    // Clear overlay
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
    }
    // Reset voice alerts and quiet observation timer
    if (activeAudioRef.current) {
      activeAudioRef.current.pause();
      activeAudioRef.current.currentTime = 0;
      activeAudioRef.current = null;
    }
    lastVoiceAlertTimeRef.current = 0;
    voiceAlertCountRef.current = 0;
    currentVoiceIncidentRef.current = null;
    normalStreakStartRef.current = null;
    setActiveVoiceNotice(null);
    setCooldownSec(0);
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  };

  // Change NTHU Demo Video
  const handleSelectDemoVideo = (filename) => {
    setSelectedDemoVideo(filename);
    fetch(`${API_BASE}/api/reset-state`, { method: 'POST' }).catch(() => {});
    if (activeAudioRef.current) {
      activeAudioRef.current.pause();
      activeAudioRef.current.currentTime = 0;
      activeAudioRef.current = null;
    }
    lastVoiceAlertTimeRef.current = 0;
    voiceAlertCountRef.current = 0;
    currentVoiceIncidentRef.current = null;
    normalStreakStartRef.current = null;
    setActiveVoiceNotice(null);
    setCooldownSec(0);
    if (isRunning && streamSource === 'nthu_demo' && videoRef.current) {
      videoRef.current.crossOrigin = 'anonymous';
      videoRef.current.srcObject = null;
      videoRef.current.src = `${API_BASE}/api/demo-video/${filename}`;
      videoRef.current.play().catch((e) => console.log('Autoplay handled:', e));
    }
  };

  // Switch Stream Source Tab
  const handleSourceTabChange = (source) => {
    setVideoError('');
    stopStream();
    fetch(`${API_BASE}/api/reset-state`, { method: 'POST' }).catch(() => {});
    setStreamSource(source);
  };

  // Trigger Personalized Baseline Calibration
  const handleTriggerCalibration = () => {
    fetch(`${API_BASE}/api/calibrate`, { method: 'POST' })
      .then((res) => res.json())
      .then((data) => {
        setIsCalibrating(true);
        let elapsed = 0;
        const interval = setInterval(() => {
          elapsed += 1;
          setCalibProgress(Math.min(100, Math.round((elapsed / 30) * 100)));
          if (elapsed >= 30) {
            clearInterval(interval);
            setIsCalibrating(false);
          }
        }, 1000);
      })
      .catch((err) => console.log('Calibration error:', err));
  };

  // Main Real-Time Frame Capture & Processing Loop
  useEffect(() => {
    let lastSendTime = 0;
    const sendIntervalMs = 70; // ~14-15 FPS network throttle for smooth inference

    const processLoop = () => {
      if (!isRunning) return;

      const now = performance.now();
      const video = videoRef.current;

      if (video && video.readyState >= 2 && video.videoWidth > 0 && !video.paused && now - lastSendTime >= sendIntervalMs && !isProcessingRef.current) {
        isProcessingRef.current = true;
        lastSendTime = now;

        try {
          const hCanvas = hiddenCanvasRef.current;
          if (hCanvas) {
            hCanvas.width = 640;
            hCanvas.height = 480;
            const hCtx = hCanvas.getContext('2d');
            hCtx.drawImage(video, 0, 0, 640, 480);
            const base64Img = hCanvas.toDataURL('image/jpeg', 0.65);

            // Post to FastAPI backend
            fetch(`${API_BASE}/api/process-frame`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ image: base64Img, source: streamSource })
            })
              .then((res) => res.json())
              .then((data) => {
                isProcessingRef.current = false;
                if (data && data.state) {
                  setTelemetry(data);

                  // Voice Alert Logic (Strict Cooldown & Max 2 Announcements per incident)
                  const nowTime = Date.now();
                  const vType = data.voice_alert_type; // 'eyes_closed' | 'yawning' | 'head_nodding' | 'distraction'

                  if (!vType || data.state === 'ALERT') {
                    if (normalStreakStartRef.current === null) {
                      normalStreakStartRef.current = nowTime;
                    }
                    // Safe / Normal state: reset alert repetition counter after 3s of continuous normal driving
                    if (nowTime - normalStreakStartRef.current >= 3000) {
                      if (currentVoiceIncidentRef.current !== null) {
                        currentVoiceIncidentRef.current = null;
                        voiceAlertCountRef.current = 0;
                        setActiveVoiceNotice(null);
                      }
                    }
                    setCooldownSec(0);
                  } else {
                    normalStreakStartRef.current = null;
                    if (currentVoiceIncidentRef.current !== vType) {
                      // New distinct alert incident started
                      currentVoiceIncidentRef.current = vType;
                      voiceAlertCountRef.current = 0;
                    }

                    const elapsedSinceLast = (nowTime - lastVoiceAlertTimeRef.current) / 1000;
                    const remainingCooldown = Math.max(0, Math.ceil(10.0 - elapsedSinceLast));
                    setCooldownSec(remainingCooldown);

                    // User rules: ensure it says two times, observe every 10 seconds after instructing
                    if (voiceAlertCountRef.current < 2) {
                      if (lastVoiceAlertTimeRef.current === 0 || elapsedSinceLast >= 10.0) {
                        voiceAlertCountRef.current += 1;
                        lastVoiceAlertTimeRef.current = nowTime;
                        speakDriverAlert(vType, voiceLanguageRef.current);
                      }
                    }
                  }

                  // Render Facial Landmarks on Overlay Canvas
                  if (showLandmarks && overlayCanvasRef.current && data.landmarks_px && data.landmarks_px.length > 0) {
                    drawLandmarksOverlay(data.landmarks_px, data.state, data.yaw, data.pitch);
                  } else if (overlayCanvasRef.current) {
                    const oCtx = overlayCanvasRef.current.getContext('2d');
                    oCtx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
                  }
                }
              })
              .catch((err) => {
                isProcessingRef.current = false;
              });
          } else {
            isProcessingRef.current = false;
          }
        } catch (e) {
          console.warn('Frame processing exception:', e);
          isProcessingRef.current = false;
        }
      }

      animationFrameRef.current = requestAnimationFrame(processLoop);
    };

    if (isRunning) {
      animationFrameRef.current = requestAnimationFrame(processLoop);
    } else {
      cancelAnimationFrame(animationFrameRef.current);
    }

    return () => cancelAnimationFrame(animationFrameRef.current);
  }, [isRunning, streamSource, isMuted, showLandmarks]);

  // Draw HUD overlay reticles and key facial landmarks
  const drawLandmarksOverlay = (pts, state, yaw, pitch) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = 640;
    canvas.height = 480;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let hudColor = '#10b981'; // Green
    if (state === 'DROWSINESS') hudColor = '#ef4444'; // Red
    if (state === 'DISTRACTION') hudColor = '#f59e0b'; // Amber

    // Eye landmarks indices in MediaPipe Face Mesh
    const eyeIndices = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380];
    const mouthIndices = [61, 81, 13, 311, 291, 402, 14, 178];

    // Draw eye contour points
    ctx.fillStyle = hudColor;
    eyeIndices.forEach((idx) => {
      if (pts[idx]) {
        ctx.beginPath();
        ctx.arc(pts[idx][0], pts[idx][1], 2.5, 0, 2 * Math.PI);
        ctx.fill();
      }
    });

    // Draw mouth contour points
    ctx.fillStyle = '#38bdf8';
    mouthIndices.forEach((idx) => {
      if (pts[idx]) {
        ctx.beginPath();
        ctx.arc(pts[idx][0], pts[idx][1], 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    });

    // Draw Nose tip reticle
    if (pts[1]) {
      ctx.strokeStyle = hudColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(pts[1][0], pts[1][1], 8, 0, 2 * Math.PI);
      ctx.stroke();

      // Head attitude vector
      ctx.beginPath();
      ctx.moveTo(pts[1][0], pts[1][1]);
      ctx.lineTo(pts[1][0] + (yaw || 0) * 2.2, pts[1][1] + (pitch || 0) * 2.2);
      ctx.stroke();
    }
  };

  // SVG Radial Gauge Calculation
  const score = Math.max(0, Math.min(100, telemetry.fatigue_score || 0));
  const circumference = 339.29;
  const strokeOffset = circumference - (score / 100) * circumference;

  const stateKey =
    telemetry.state === 'DROWSINESS' ? 'drowsiness' : telemetry.state === 'DISTRACTION' ? 'distraction' : 'alert';
  const ta = voiceLanguage === 'ta';

  return (
    <div className="app-container">
      <canvas ref={hiddenCanvasRef} style={{ display: 'none' }} />

      {/* TOP BAR */}
      <header className="top-bar">
        <div className="top-title">
          <Activity size={22} color="#2563eb" />
          <div>
            <div className="top-title-main">Driver Fatigue &amp; Distraction Monitor</div>
            <div className="top-title-sub">NTHU-DDD • 15s sliding window • Real-time ADAS</div>
          </div>
        </div>

        <div className="top-actions">
          <div
            className="lang-toggle-button"
            role="group"
            aria-label="Voice alert language"
          >
            <button
              type="button"
              className={`lang-toggle-option ${voiceLanguage === 'en' ? 'active' : ''}`}
              onClick={() => setVoiceLanguage('en')}
            >
              English
            </button>
            <button
              type="button"
              className={`lang-toggle-option ${voiceLanguage === 'ta' ? 'active' : ''}`}
              onClick={() => setVoiceLanguage('ta')}
            >
              தமிழ்
            </button>
          </div>

          <button
            onClick={() => setIsMuted(!isMuted)}
            className="btn-secondary"
            title={isMuted ? 'Unmute Audio Alarm' : 'Mute Audio Alarm'}
          >
            {isMuted ? <VolumeX size={16} color="#dc2626" /> : <Volume2 size={16} color="#059669" />}
            <span>{isMuted ? 'Muted' : 'Sound ON'}</span>
          </button>

          <button onClick={() => setShowMetricsModal(true)} className="btn-secondary">
            <BarChart3 size={16} color="#2563eb" />
            <span>Model Metrics</span>
          </button>

          <div className={`status-pill ${isRunning ? 'on' : ''}`}>
            <span>●</span>
            <span>{isRunning ? 'STREAMING' : 'STANDBY'}</span>
          </div>
        </div>
      </header>

      {/* MAIN GRID — fills the remaining viewport height, no page scroll */}
      <div className="dashboard-main-grid">
        {/* LEFT: controls on top, video fills the rest */}
        <section className="panel left-panel">
          <div className="controls-row">
            <div className="source-tabs">
              <button
                onClick={() => handleSourceTabChange('webcam')}
                className={streamSource === 'webcam' ? 'btn-primary' : 'btn-secondary'}
              >
                <Camera size={16} />
                <span>Live Webcam</span>
              </button>
              <button
                onClick={() => handleSourceTabChange('nthu_demo')}
                className={streamSource === 'nthu_demo' ? 'btn-primary' : 'btn-secondary'}
              >
                <Film size={16} />
                <span>NTHU Videos</span>
              </button>
            </div>

            <div className="action-btn-row">
              {!isRunning ? (
                <button onClick={startStream} className="btn-start">
                  <Play size={16} />
                  <span>Start Monitoring</span>
                </button>
              ) : (
                <button onClick={stopStream} className="btn-danger">
                  <Square size={16} />
                  <span>Stop Monitoring</span>
                </button>
              )}
              <button onClick={handleTriggerCalibration} className="btn-secondary" disabled={isCalibrating}>
                <Crosshair size={16} color="#2563eb" />
                <span>{isCalibrating ? `Calibrating (${calibProgress}%)` : '30s Calibration'}</span>
              </button>
              <label className="landmark-switch" htmlFor="chk-landmarks">
                <input
                  type="checkbox"
                  id="chk-landmarks"
                  checked={showLandmarks}
                  onChange={(e) => setShowLandmarks(e.target.checked)}
                />
                <span>Landmarks</span>
              </label>
            </div>
          </div>

          {streamSource === 'nthu_demo' && (
            <div className="nthu-strip">
              {demoVideos.map((vid) => (
                <button
                  key={vid.id}
                  onClick={() => handleSelectDemoVideo(vid.filename)}
                  className={`nthu-chip ${selectedDemoVideo === vid.filename ? 'active' : ''}`}
                  title={`${vid.description} — Expected: ${vid.expected_state}`}
                >
                  {vid.title}
                </button>
              ))}
            </div>
          )}

          {(videoError || (!backendOk && streamSource === 'nthu_demo')) && (
            <div className="error-banner">
              {videoError || `Backend not reachable at ${API_BASE}. Start it with backend/run_backend.bat, then press Start Monitoring.`}
            </div>
          )}

          <div className="video-stage">
            <div className="video-wrapper">
              <video
                ref={videoRef}
                className="video-elem"
                crossOrigin="anonymous"
                playsInline
                muted
                autoPlay
                loop={streamSource === 'nthu_demo'}
                onError={() => {
                  if (streamSource === 'nthu_demo' && isRunning) {
                    setVideoError('Could not load the demo video. Make sure the backend is running at ' + API_BASE);
                    setIsRunning(false);
                  }
                }}
              />
              <canvas ref={overlayCanvasRef} className="overlay-canvas" />
              <div className="video-tag video-tag-left">
                {streamSource === 'webcam' ? 'WEBCAM' : 'NTHU-DDD CLIP'}
              </div>
              {telemetry.face_detected && <div className="video-tag video-tag-right">FACE LOCKED</div>}
            </div>
          </div>
        </section>

        {/* RIGHT: status, gauge, metrics */}
        <section className="right-col">
          <div className={`status-card ${stateKey}`}>
            <div className="status-badge-icon">
              {telemetry.state === 'DROWSINESS' ? (
                <AlertOctagon size={28} />
              ) : telemetry.state === 'DISTRACTION' ? (
                <AlertTriangle size={28} />
              ) : (
                <ShieldCheck size={28} />
              )}
            </div>
            <div className="status-text">
              <div className="status-content-title">
                {telemetry.state === 'DROWSINESS'
                  ? 'CRITICAL: DROWSINESS'
                  : telemetry.state === 'DISTRACTION'
                  ? 'WARNING: DISTRACTION'
                  : 'STATUS: NORMAL (ALERT)'}
              </div>
              <div className="status-content-desc">
                {ta
                  ? telemetry.buzzer_reason_ta || telemetry.buzzer_reason || 'இயல்பான விழிப்புணர்வுடன் பாதுகாப்பாக ஓட்டுதல்.'
                  : telemetry.buzzer_reason || 'Driver attentive, eyes open, facing forward.'}
              </div>

              {telemetry.is_head_turned && telemetry.distraction_duration > 0 && telemetry.distraction_duration < 5.0 && (
                <div className="status-chip warn">
                  <Clock size={13} />
                  <span>
                    {ta
                      ? `தலை திரும்பியுள்ளது (${Number(telemetry.distraction_duration).toFixed(1)}வி / 5.0வி)`
                      : `Head turned: observing ${Number(telemetry.distraction_duration).toFixed(1)}s / 5.0s`}
                  </span>
                </div>
              )}

              {telemetry.buzzer && (
                <div className="status-chip-row">
                  <div className="status-chip">
                    <Volume2 size={13} />
                    <span>{ta ? telemetry.buzzer_reason_ta || telemetry.buzzer_reason : telemetry.buzzer_reason}</span>
                  </div>
                  <div className={`status-chip ${cooldownSec > 0 ? 'warn' : 'ok'}`}>
                    <Clock size={13} />
                    <span>
                      {cooldownSec > 0
                        ? ta ? `இயக்கி கவனிப்பு (${cooldownSec}வி)` : `Observing (${cooldownSec}s)`
                        : voiceAlertCountRef.current >= 2
                        ? ta ? 'அதிகபட்ச 2 எச்சரிக்கைகள்' : 'Max 2 alerts reached'
                        : ta ? 'குரல் தயார்' : 'Voice ready'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Fatigue gauge */}
          <div className="panel gauge-card">
            <svg width="96" height="96" viewBox="0 0 140 140">
              <circle cx="70" cy="70" r="54" fill="none" stroke="#e2e8f0" strokeWidth="11" />
              <circle
                cx="70"
                cy="70"
                r="54"
                fill="none"
                stroke={score > 60 ? '#dc2626' : score > 35 ? '#d97706' : '#059669'}
                strokeWidth="11"
                strokeDasharray={circumference}
                strokeDashoffset={strokeOffset}
                strokeLinecap="round"
                transform="rotate(-90 70 70)"
                style={{ transition: 'stroke-dashoffset 0.4s ease' }}
              />
              <text x="70" y="78" textAnchor="middle" fontSize="30" fontWeight="800" fill="#0f172a">
                {Math.round(score)}%
              </text>
            </svg>
            <div>
              <div className="gauge-label">Fatigue Score</div>
              <div
                className="gauge-trend"
                style={{ color: telemetry.fatigue_trend === 'Increasing' ? '#dc2626' : '#059669' }}
              >
                {telemetry.fatigue_trend === 'Increasing' ? '📈' : '📉'} {telemetry.fatigue_trend}
              </div>
              <div className="gauge-sub">15s sliding analysis</div>
            </div>
          </div>

          {/* Telemetry cards */}
          <div className="telemetry-grid">
            <div className="panel telemetry-card">
              <div className="telemetry-card-title">
                <Eye size={14} color="#2563eb" />
                <span>Eye Aspect Ratio</span>
              </div>
              <div className="telemetry-card-val">{telemetry.ear?.toFixed(2) || '0.28'}</div>
              <div className="telemetry-card-sub">
                <span>Mean: {telemetry.EAR_mean?.toFixed(2)}</span>
                <span>Min: {telemetry.EAR_min?.toFixed(2)}</span>
              </div>
            </div>

            <div className="panel telemetry-card">
              <div className="telemetry-card-title">
                <Clock size={14} color="#dc2626" />
                <span>PERCLOS (% Closure)</span>
              </div>
              <div
                className="telemetry-card-val"
                style={{ color: telemetry.perclos > 0.16 ? '#dc2626' : '#059669' }}
              >
                {((telemetry.perclos || 0) * 100).toFixed(1)}%
              </div>
              <div className="telemetry-card-sub">
                <span>Threshold: 16.0%</span>
                <span>Window: 15s</span>
              </div>
            </div>

            <div className="panel telemetry-card">
              <div className="telemetry-card-title">
                <Activity size={14} color="#059669" />
                <span>Blinks &amp; Duration</span>
              </div>
              <div className="telemetry-card-val">
                {telemetry.blink_frequency?.toFixed(1) || '16.0'} <span className="unit">/min</span>
              </div>
              <div className="telemetry-card-sub">
                <span>Avg: {telemetry.avg_blink_duration?.toFixed(2)}s</span>
                <span>Total: {telemetry.blink_count || 0}</span>
              </div>
            </div>

            <div className="panel telemetry-card">
              <div className="telemetry-card-title">
                <span>👄 Mouth Aspect Ratio</span>
              </div>
              <div className="telemetry-card-val">{telemetry.mar?.toFixed(2) || '0.25'}</div>
              <div className="telemetry-card-sub">
                <span>Yawn Count: {telemetry.yawn_count || 0}</span>
                <span>Threshold: 0.58</span>
              </div>
            </div>

            <div className="panel telemetry-card">
              <div className="telemetry-card-title">
                <Compass size={14} color="#d97706" />
                <span>Head Pose Orientation</span>
              </div>
              <div className="telemetry-card-val">
                {telemetry.pitch?.toFixed(1) || '0.0'}° <span className="unit">Pitch</span>
              </div>
              <div className="telemetry-card-sub">
                <span>Yaw: {telemetry.yaw?.toFixed(1)}°</span>
                <span>Nods: {telemetry.head_nod_count || 0}</span>
              </div>
            </div>

            <div className="panel telemetry-card">
              <div className="telemetry-card-title">
                <Crosshair size={14} color="#2563eb" />
                <span>Gaze Direction</span>
              </div>
              <div className="telemetry-card-val gaze">{telemetry.gaze_direction || 'Center / Forward'}</div>
              <div className="telemetry-card-sub">
                <span>Stability: {Math.round((telemetry.gaze_stability || 0) * 100)}%</span>
              </div>
            </div>
          </div>

          {/* Baseline strip */}
          <div className="panel baseline-strip">
            <div>
              🎯 <strong>Baseline</strong> EAR:{' '}
              <code>{telemetry.baseline_summary?.baseline_ear?.toFixed(2)}</code> • Blink:{' '}
              <code>{telemetry.baseline_summary?.baseline_blink_duration?.toFixed(2)}s</code>
            </div>
            <div>
              EAR Dev:{' '}
              <strong>
                {telemetry.deviations?.EAR_deviation_from_baseline > 0 ? '+' : ''}
                {telemetry.deviations?.EAR_deviation_from_baseline?.toFixed(2)}
              </strong>
            </div>
          </div>
        </section>
      </div>

      {/* METRICS & DOCUMENTATION MODAL */}
      {showMetricsModal && (
        <div
          onClick={() => setShowMetricsModal(false)}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15,23,42,0.45)',
            backdropFilter: 'blur(4px)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px',
            overflowY: 'auto'
          }}
        >
          <div
            className="panel"
            onClick={(e) => e.stopPropagation()}
            style={{
              maxWidth: '900px',
              width: '100%',
              maxHeight: '88vh',
              overflowY: 'auto',
              padding: '26px',
              position: 'relative'
            }}
          >
            <button
              onClick={() => setShowMetricsModal(false)}
              style={{
                position: 'absolute',
                top: '18px',
                right: '18px',
                background: 'none',
                border: 'none',
                color: '#64748b',
                cursor: 'pointer'
              }}
            >
              <X size={24} />
            </button>

            <h2 style={{ color: '#0f172a', marginBottom: '8px' }}>
              📊 NTHU-DDD Model Evaluation & Benchmark Comparison
            </h2>
            <p style={{ fontSize: '0.88rem', color: '#475569', marginBottom: '16px' }}>
              Rigorous evaluation conducted on held-out test subjects using <strong>Subject-Aware Partitioning</strong>{' '}
              to guarantee zero subject identity data leakage.
            </p>

            {modelMetrics && (
              <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '20px', fontSize: '0.9rem' }}>
                <thead>
                  <tr style={{ background: '#f1f5f9', textAlign: 'left', color: '#1d4ed8' }}>
                    <th style={{ padding: '10px' }}>Model</th>
                    <th style={{ padding: '10px' }}>Accuracy</th>
                    <th style={{ padding: '10px' }}>Precision</th>
                    <th style={{ padding: '10px' }}>Recall</th>
                    <th style={{ padding: '10px' }}>F1-Score</th>
                    <th style={{ padding: '10px' }}>ROC-AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {modelMetrics.comparison_table.map((row, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '10px', fontWeight: 700, color: '#0f172a' }}>{row.Model}</td>
                      <td style={{ padding: '10px' }}>{row.Accuracy}</td>
                      <td style={{ padding: '10px' }}>{row.Precision}</td>
                      <td style={{ padding: '10px' }}>{row.Recall}</td>
                      <td style={{ padding: '10px' }}>{row['F1-Score']}</td>
                      <td style={{ padding: '10px' }}>{row['ROC-AUC']}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: '16px' }}>
              <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', padding: '12px', borderRadius: '8px' }}>
                <h4 style={{ color: '#1d4ed8', marginBottom: '8px' }}>Confusion Matrix (Test Subjects)</h4>
                <img
                  src={`${API_BASE}/api/results/all_confusion_matrices.png`}
                  alt="Confusion Matrix"
                  style={{ width: '100%', borderRadius: '6px' }}
                />
              </div>

              <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', padding: '12px', borderRadius: '8px' }}>
                <h4 style={{ color: '#059669', marginBottom: '8px' }}>ROC Curves Comparison</h4>
                <img
                  src={`${API_BASE}/api/results/roc_curves.png`}
                  alt="ROC Curves"
                  style={{ width: '100%', borderRadius: '6px' }}
                />
              </div>
            </div>

            <div style={{ marginTop: '20px', textAlign: 'right' }}>
              <button onClick={() => setShowMetricsModal(false)} className="btn-primary">
                Close Benchmark Window
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
