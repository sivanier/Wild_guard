# Animal Intrusion Detection & Crop Protection System

Flask serves the field detection UI and authenticated Farmer Dashboard. YOLO inference uses `models/best.pt` when present (otherwise the existing COCO `yolov8n.pt` fallback); SQLite stores detection history and analytics. Flask-SocketIO now broadcasts each newly grouped detection to logged-in dashboard browsers in real time. Both laptops connect directly over the private LAN; there is no cloud service.

## Windows setup and run (Laptop A)

Open PowerShell in this project directory:

```powershell
cd C:\Users\Admin\Downloads\animal_intrusion_system\animal_intrusion_system
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:FLASK_SECRET_KEY = (python -c "import secrets; print(secrets.token_hex(32))")
python app.py
```

If PowerShell blocks activation, for this terminal only run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. The server listens on `0.0.0.0:5000` by default. On Laptop A open `http://127.0.0.1:5000/` and use its image, video, or webcam workflow. The existing alarm remains active.

## Connect Laptop B

1. Put both laptops on the same trusted Wi-Fi/LAN. On Laptop A run `ipconfig` and find the active Wi-Fi/Ethernet adapter's `IPv4 Address` (for example, `192.168.1.42`). Do not use `127.0.0.1` on Laptop B.
2. If Windows Firewall prompts, allow Python on **Private networks only**. If there is no prompt and access is blocked, open PowerShell as Administrator on Laptop A and add the narrowly scoped TCP rule:

   ```powershell
   New-NetFirewallRule -DisplayName "Animal Intrusion Dashboard (Private LAN)" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 5000 -Profile Private -RemoteAddress LocalSubnet
   ```

   Remove the rule when no longer needed with `Remove-NetFirewallRule -DisplayName "Animal Intrusion Dashboard (Private LAN)"`. Keep the Wi-Fi network profile Private; do not create a Public-profile or router port-forward rule.

3. On Laptop B open `http://<Laptop-A-IPv4>:5000/farmer`, replacing the placeholder with the address from `ipconfig` (for example `http://192.168.1.42:5000/farmer`). Log in with the demo account `farmer` / `farmer123`.
4. The dashboard shows its live connection state. A detected animal produces a notification containing the animal, confidence, timestamp, source, and box coordinates when available. History and analytics reload automatically. Socket.IO reconnects after temporary disconnections and reloads history after reconnecting.

The dashboard loads the Socket.IO browser client from the Socket.IO CDN, so Laptop B needs internet access to load that script on its first page load. Detection traffic and data remain on the LAN. If a network blocks that CDN, the live client cannot start; allow the CDN or host a pinned Socket.IO client file locally.

## Detection grouping and storage

Every detection row is persisted in the existing `detections.db`. For webcam and uploaded-video sources, repeated detections of the same animal/source within 8 seconds are grouped to prevent consecutive-frame alert spam; image uploads and webcam still-image requests remain separate events. The newest database schema adds an optional bounding-box column through a safe startup migration. Crop damage and risk remain the existing illustrative estimates and are not validated agricultural predictions.

## Checks

Install dependencies with the commands above. `python app.py` starts the LAN service. Dashboard access still requires the existing demo login. Replace the demo credentials and set a private `FLASK_SECRET_KEY` before using this beyond a controlled project demonstration. Do not expose port 5000 to the public internet.
