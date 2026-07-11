let data = null;
let sampleIndex = 0;
let frame = 0;
let timer = null;

const realCanvas = document.getElementById("real");
const imaginedCanvas = document.getElementById("imagined");
const driftCanvas = document.getElementById("drift");
const summary = document.getElementById("summary");
const realStats = document.getElementById("realStats");
const imaginedStats = document.getElementById("imaginedStats");

document.getElementById("prev").addEventListener("click", () => changeSample(-1));
document.getElementById("next").addEventListener("click", () => changeSample(1));
document.getElementById("play").addEventListener("click", togglePlay);

fetch("/api/rollouts")
  .then((response) => response.json())
  .then((payload) => {
    if (typeof payload === "string") {
      payload = JSON.parse(payload);
    }
    if (payload.error) {
      summary.textContent = payload.error;
      return;
    }
    data = payload;
    const s = data.summary;
    summary.textContent = `Real return ${s.mean_real_return.toFixed(3)} · imagined return ${s.mean_imagined_return.toFixed(3)} · gap ${s.mean_return_gap.toFixed(3)}`;
    render();
  })
  .catch((error) => {
    summary.textContent = `Could not load rollouts: ${error}`;
  });

function changeSample(delta) {
  if (!data) return;
  sampleIndex = (sampleIndex + delta + data.samples.length) % data.samples.length;
  frame = 0;
  render();
}

function togglePlay() {
  const button = document.getElementById("play");
  if (timer) {
    clearInterval(timer);
    timer = null;
    button.textContent = "Play";
    return;
  }
  timer = setInterval(() => {
    const sample = data.samples[sampleIndex];
    const maxFrames = Math.max(sample.real.states.length, sample.imagined.states.length);
    frame = (frame + 1) % maxFrames;
    render();
  }, 350);
  button.textContent = "Pause";
}

function render() {
  if (!data) return;
  const sample = data.samples[sampleIndex];
  drawGrid(realCanvas, data.grid, sample.real.states, frame, "#1f77b4");
  drawGrid(imaginedCanvas, data.grid, sample.imagined.states, frame, "#d95f02");
  drawDrift(driftCanvas, data.drift);
  realStats.textContent = `Sample ${sampleIndex + 1}: return ${sample.real.return.toFixed(3)}, length ${sample.real.length}`;
  imaginedStats.textContent = `Sample ${sampleIndex + 1}: return ${sample.imagined.return.toFixed(3)}, length ${sample.imagined.length}`;
}

function drawGrid(canvas, grid, states, frameIndex, color) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const cell = Math.min(w / grid.width, h / grid.height);
  const offsetX = (w - cell * grid.width) / 2;
  const offsetY = (h - cell * grid.height) / 2;

  ctx.strokeStyle = "#d8dee2";
  for (let y = 0; y < grid.height; y += 1) {
    for (let x = 0; x < grid.width; x += 1) {
      ctx.strokeRect(offsetX + x * cell, offsetY + y * cell, cell, cell);
    }
  }

  ctx.fillStyle = "#263238";
  for (const wall of grid.walls) {
    ctx.fillRect(offsetX + wall[0] * cell, offsetY + wall[1] * cell, cell, cell);
  }

  ctx.fillStyle = "#2f9e44";
  ctx.fillRect(offsetX + grid.goal[0] * cell + 4, offsetY + grid.goal[1] * cell + 4, cell - 8, cell - 8);

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  const visible = states.slice(0, Math.min(frameIndex + 1, states.length));
  visible.forEach((state, idx) => {
    const p = decode(state, grid);
    const cx = offsetX + (p.x + 0.5) * cell;
    const cy = offsetY + (p.y + 0.5) * cell;
    if (idx === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  });
  ctx.stroke();

  const current = decode(states[Math.min(frameIndex, states.length - 1)], grid);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(offsetX + (current.x + 0.5) * cell, offsetY + (current.y + 0.5) * cell, Math.max(7, cell * 0.18), 0, Math.PI * 2);
  ctx.fill();
}

function decode(state, grid) {
  return {
    x: Math.max(0, Math.min(grid.width - 1, Math.round(state[0] * (grid.width - 1)))),
    y: Math.max(0, Math.min(grid.height - 1, Math.round(state[1] * (grid.height - 1)))),
  };
}

function drawDrift(canvas, drift) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#d8dee2";
  ctx.strokeRect(40, 20, canvas.width - 60, canvas.height - 50);
  if (!drift || drift.length < 2) return;
  const maxValue = Math.max(...drift, 0.001);
  ctx.strokeStyle = "#c43c39";
  ctx.lineWidth = 3;
  ctx.beginPath();
  drift.forEach((value, idx) => {
    const x = 40 + (idx / (drift.length - 1)) * (canvas.width - 60);
    const y = 20 + (1 - value / maxValue) * (canvas.height - 50);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#4e5b64";
  ctx.font = "14px system-ui";
  ctx.fillText("mean state error over imagined horizon", 44, canvas.height - 12);
}
