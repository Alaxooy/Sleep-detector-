import io

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from drowsiness_detector import DrowsinessDetector

app = FastAPI(title="Drowsiness Detection API")
detector = DrowsinessDetector()


@app.on_event("startup")
async def startup_event():
    detector.start()


@app.on_event("shutdown")
async def shutdown_event():
    detector.stop()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/status")
async def status():
    return detector.get_status()


@app.post("/start")
async def start():
    detector.start()
    return {"status": "started"}


@app.post("/stop")
async def stop():
    detector.stop()
    return {"status": "stopped"}


@app.get("/snapshot")
async def snapshot():
    image_bytes = detector.get_snapshot()
    if image_bytes is None:
        raise HTTPException(status_code=503, detail="No snapshot available")
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/jpeg")


@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html lang='en'>
    <head>
      <meta charset='UTF-8' />
      <meta name='viewport' content='width=device-width, initial-scale=1.0' />
      <title>Drowsiness Detection Dashboard</title>
      <style>
        :root {
          color-scheme: dark;
          --bg: #0f1220;
          --surface: rgba(24, 30, 60, 0.95);
          --surface-strong: #131a39;
          --border: rgba(255, 255, 255, 0.08);
          --text: #f7f8ff;
          --muted: #a5b1d5;
          --primary: #7c5cff;
          --accent: #3c9dff;
          --danger: #ff5c7a;
          --success: #4ee5b8;
          --shadow: 0 30px 80px rgba(0, 0, 0, 0.25);
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          min-height: 100vh;
          font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          color: var(--text);
          background: radial-gradient(circle at top left, rgba(124, 92, 255, 0.18), transparent 28%),
                      radial-gradient(circle at bottom right, rgba(60, 157, 255, 0.12), transparent 30%),
                      linear-gradient(180deg, #0b0d18 0%, #10142b 100%);
        }
        .page {
          width: min(1240px, calc(100% - 40px));
          margin: 0 auto;
          padding: 32px 0;
        }
        header {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 28px;
        }
        .hero {
          max-width: 640px;
        }
        .hero h1 {
          margin: 0;
          font-size: clamp(2.1rem, 3vw, 3.6rem);
          line-height: 1.02;
          letter-spacing: -0.04em;
        }
        .hero p {
          margin: 18px 0 0;
          color: var(--muted);
          font-size: 1rem;
          line-height: 1.8;
          max-width: 560px;
        }
        .badge {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 8px 14px;
          border-radius: 999px;
          background: rgba(255,255,255,0.08);
          border: 1px solid rgba(255,255,255,0.09);
          color: var(--primary);
          font-size: 0.92rem;
          font-weight: 600;
        }
        .grid {
          display: grid;
          grid-template-columns: 1.6fr 1fr;
          gap: 22px;
          align-items: start;
        }
        .panel {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 24px;
          box-shadow: var(--shadow);
          padding: 24px;
          backdrop-filter: blur(18px);
        }
        .panel h2 {
          margin: 0 0 18px;
          font-size: 1.2rem;
          letter-spacing: -0.02em;
        }
        .preview {
          position: relative;
          min-height: 420px;
          border-radius: 22px;
          overflow: hidden;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(180deg, rgba(18,22,44,0.8), rgba(10,12,24,0.96));
        }
        .preview img {
          width: 100%;
          height: 100%;
          object-fit: contain;
          display: block;
          background: #090b16;
        }
        .status-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 14px;
        }
        .status-card {
          background: #12192f;
          border: 1px solid rgba(255,255,255,0.06);
          border-radius: 18px;
          padding: 18px 20px;
        }
        .status-card span {
          display: block;
          margin-top: 8px;
          font-size: 1.35rem;
          font-weight: 700;
          color: var(--text);
        }
        .status-card small {
          color: var(--muted);
        }
        .status-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 16px;
          margin-bottom: 18px;
          flex-wrap: wrap;
        }
        .pill {
          display: inline-flex;
          align-items: center;
          padding: 10px 16px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.04);
          color: var(--text);
          font-weight: 600;
        }
        .pill.safe { color: var(--success); border-color: rgba(78,229,184,0.24); }
        .pill.alert { color: var(--danger); border-color: rgba(255,92,122,0.24); }
        .actions {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          margin-top: 14px;
        }
        .button {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          padding: 12px 20px;
          border-radius: 14px;
          border: none;
          font-size: 0.98rem;
          font-weight: 700;
          cursor: pointer;
          transition: transform 0.18s ease, background 0.2s ease, box-shadow 0.2s ease;
        }
        .button:hover { transform: translateY(-1px); }
        .button.primary { background: linear-gradient(135deg, var(--primary), var(--accent)); color: #fff; box-shadow: 0 18px 36px rgba(124,92,255,0.2); }
        .button.secondary { background: rgba(255,255,255,0.04); color: var(--text); }
        .button.danger { background: rgba(255,92,122,0.18); color: var(--danger); }
        .footer {
          padding-top: 26px;
          color: var(--muted);
          font-size: 0.95rem;
        }
        a { color: var(--accent); text-decoration: none; }
        a:hover { text-decoration: underline; }
        @media (max-width: 900px) {
          .grid { grid-template-columns: 1fr; }
        }
      </style>
    </head>
    <body>
      <div class='page'>
        <header>
          <div class='hero'>
            <span class='badge'>Live Drowsiness Monitor</span>
            <h1>Drowsiness detection built for clarity and control.</h1>
            <p>Monitor your eyes in real time, get visual feedback, and keep the service running from one polished dashboard.</p>
          </div>
          <div class='pill safe'>Ready for monitoring</div>
        </header>

        <div class='grid'>
          <section class='panel'>
            <div class='status-row'>
              <div>
                <h2>Live Preview</h2>
                <p style='color: var(--muted); margin: 8px 0 0;'>Latest annotated frame from the camera feed.</p>
              </div>
              <div class='actions'>
                <button class='button primary' onclick='refreshSnapshot()'>Refresh</button>
                <button class='button secondary' onclick='fetchAction("/status")'>Status</button>
              </div>
            </div>
            <div class='preview'>
              <img id='snapshot' src='/snapshot' alt='Live camera snapshot' />
            </div>
          </section>

          <aside class='panel'>
            <div class='status-card'>
              <small>Current mode</small>
              <span id='running'>Loading...</span>
            </div>
            <div class='status-card'>
              <small>Eye Aspect Ratio</small>
              <span id='ear'>Loading...</span>
            </div>
            <div class='status-card'>
              <small>Alert state</small>
              <span id='alert' class='pill'>Loading...</span>
            </div>
            <div class='status-card'>
              <small>Closed frames</small>
              <span id='closed_frames'>Loading...</span>
            </div>
            <div class='actions' style='margin-top: 20px; width: 100%;'>
              <button class='button primary' onclick='postAction("/start")'>Start</button>
              <button class='button danger' onclick='postAction("/stop")'>Stop</button>
            </div>
            <div class='footer'>
              <p>Use the buttons above to control camera monitoring or visit <a href='/docs'>API docs</a> for integrations.</p>
            </div>
          </aside>
        </div>
      </div>

      <script>
        async function updateStatus() {
          try {
            const response = await fetch('/status');
            const data = await response.json();
            document.getElementById('running').textContent = data.running ? 'Running' : 'Stopped';
            document.getElementById('ear').textContent = data.ear.toFixed(2);
            const alertEl = document.getElementById('alert');
            alertEl.textContent = data.alert ? 'ACTIVE' : 'Safe';
            alertEl.className = 'pill ' + (data.alert ? 'alert' : 'safe');
            document.getElementById('closed_frames').textContent = data.closed_frames;
          } catch (err) {
            console.error(err);
          }
        }

        async function refreshSnapshot() {
          const img = document.getElementById('snapshot');
          img.src = '/snapshot?ts=' + new Date().getTime();
        }

        async function postAction(path) {
          try {
            await fetch(path, { method: 'POST' });
            await updateStatus();
          } catch (err) {
            console.error(err);
          }
        }

        async function fetchAction(path) {
          await postAction(path);
          refreshSnapshot();
        }

        setInterval(updateStatus, 500);
        setInterval(refreshSnapshot, 100);
        updateStatus();
      </script>
    </body>
    </html>
    """
    return html_content


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
