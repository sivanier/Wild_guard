// One Socket.IO client per page, shared by the field station and farmer dashboard.
(function () {
  if (window.detectionAlertsInitialized) return;
  window.detectionAlertsInitialized = true;

  const status = document.getElementById('connectionStatus');
  const toastContainer = document.getElementById('toastContainer');
  const soundToggle = document.getElementById('alertSoundToggle');
  const seenIds = new Set();
  let soundEnabled = false;

  function setStatus(message, state) {
    if (!status) return;
    status.textContent = message;
    status.className = `connection-status is-${state}`;
  }

  if (soundToggle) {
    soundToggle.addEventListener('click', () => {
      soundEnabled = !soundEnabled;
      soundToggle.setAttribute('aria-pressed', String(soundEnabled));
      soundToggle.textContent = soundEnabled ? 'Mute alert sound' : 'Enable alert sound';
    });
  }

  function dismissToast(toast) {
    if (toast.dismissTimer) window.clearTimeout(toast.dismissTimer);
    toast.classList.add('toast-leaving');
    window.setTimeout(() => toast.remove(), 180);
  }

  function showToast(detection) {
    if (!toastContainer) return;
    const allowedSeverities = new Set(['high', 'medium', 'low']);
    const severity = allowedSeverities.has(detection.risk_severity)
      ? detection.risk_severity : 'low';
    const highRisk = severity === 'high';
    const toast = document.createElement('section');
    toast.className = `detection-toast severity-${severity}${highRisk ? ' is-prominent' : ''}`;
    toast.setAttribute('role', highRisk ? 'alert' : 'status');
    toast.setAttribute('aria-atomic', 'true');

    const icon = document.createElement('span');
    icon.className = 'toast-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = highRisk ? '⚠' : '●';

    const content = document.createElement('div');
    content.className = 'toast-content';
    const heading = document.createElement('strong');
    heading.textContent = highRisk ? 'High risk animal detected' : 'Animal detected';
    const animal = document.createElement('span');
    animal.className = 'toast-animal';
    animal.textContent = `${detection.animal} · ${(Number(detection.confidence) * 100).toFixed(1)}% confidence`;
    const metadata = document.createElement('span');
    metadata.className = 'toast-metadata';
    const date = new Date(detection.timestamp);
    const timestamp = Number.isNaN(date.getTime()) ? detection.timestamp : date.toLocaleString();
    metadata.textContent = `${detection.source} · ${timestamp} · ${severity} risk`;
    content.append(heading, animal, metadata);

    const close = document.createElement('button');
    close.className = 'toast-dismiss';
    close.type = 'button';
    close.setAttribute('aria-label', 'Dismiss animal detection alert');
    close.textContent = '×';
    close.addEventListener('click', () => dismissToast(toast));
    toast.append(icon, content, close);
    toastContainer.appendChild(toast);

    while (toastContainer.children.length > 4) {
      const oldest = toastContainer.firstElementChild;
      if (oldest.dismissTimer) window.clearTimeout(oldest.dismissTimer);
      oldest.remove();
    }
    toast.dismissTimer = window.setTimeout(() => dismissToast(toast), highRisk ? 20000 : 8000);

    if (highRisk && soundEnabled && typeof window.playAlarm === 'function') {
      try { window.playAlarm(); } catch (error) { console.warn('Alert sound was blocked by the browser.', error); }
    }
  }

  if (typeof window.io !== 'function') {
    setStatus('Live alerts unavailable: Socket.IO client did not load.', 'error');
    return;
  }

  try {
    const namespace = window.DETECTION_SOCKET_NAMESPACE || '/field';
    const socket = window.io(namespace, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      timeout: 10000
    });

    socket.on('connect', () => setStatus('Connected · live alerts enabled', 'connected'));
    socket.on('disconnect', () => setStatus('Disconnected · reconnecting…', 'disconnected'));
    socket.on('connect_error', error => {
      setStatus(`Connection error · retrying (${error.message})`, 'error');
    });
    socket.on('detection', detection => {
      const id = detection && detection.id;
      if (id === undefined || id === null) return;
      const key = String(id);
      if (seenIds.has(key)) return;
      seenIds.add(key);
      if (seenIds.size > 500) seenIds.delete(seenIds.values().next().value);

      showToast(detection);
      window.dispatchEvent(new CustomEvent('detection-recorded', { detail: detection }));
    });
  } catch (error) {
    setStatus(`Live alerts could not connect: ${error.message}`, 'error');
  }
})();
