// Minimal player for silent explainer videos: transport sits BELOW the frame so it never covers the on-screen captions.
// player(v, {start}) -> {el, video, seek(t), onTime(fn)}; styling through CSS custom properties on .pl (see each page).
(function () {
  const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const ICON = {
    play: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M6 4.2v11.6a.6.6 0 0 0 .9.5l9.3-5.8a.6.6 0 0 0 0-1L6.9 3.7a.6.6 0 0 0-.9.5Z" fill="currentColor"/></svg>',
    pause: '<svg viewBox="0 0 20 20" aria-hidden="true"><rect x="5" y="4" width="3.4" height="12" rx=".8" fill="currentColor"/><rect x="11.6" y="4" width="3.4" height="12" rx=".8" fill="currentColor"/></svg>',
    full: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3.5 7.5v-4h4M12.5 3.5h4v4M16.5 12.5v4h-4M7.5 16.5h-4v-4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  };
  const all = new Set();
  window.fmt = fmt;
  window.player = function (v, opt = {}) {
    const el = document.createElement("div"); el.className = "pl";
    el.innerHTML = `
      <div class="pl-frame"><video class="pl-video" preload="metadata" playsinline muted poster="${v.poster}" src="${v.file}"
        aria-label="Video ${v.n}: ${v.title}, silent, ${fmt(v.duration)}"></video>
        <button class="pl-big" aria-label="Play video ${v.n}">${ICON.play}</button></div>
      <div class="pl-bar">
        <button class="pl-btn pl-pp" aria-label="Play">${ICON.play}</button>
        <div class="pl-scrub"><div class="pl-ticks">${v.chapters.map(c => `<i style="left:${(100 * c.t / v.duration).toFixed(2)}%" title="${c.text}"></i>`).join("")}</div>
          <div class="pl-fill"></div>
          <input type="range" min="0" max="${v.duration}" step="0.1" value="0" aria-label="Seek in video ${v.n}"></div>
        <span class="pl-time"><span class="pl-now">0:00</span> / ${fmt(v.duration)}</span>
        <button class="pl-btn pl-fs" aria-label="Full screen">${ICON.full}</button>
      </div>`;
    const video = el.querySelector("video"), pp = el.querySelector(".pl-pp"), big = el.querySelector(".pl-big"),
      range = el.querySelector("input"), fill = el.querySelector(".pl-fill"), now = el.querySelector(".pl-now");
    const subs = [];
    const toggle = () => (video.paused ? video.play() : video.pause());
    const sync = () => {
      const t = video.currentTime; range.value = t; now.textContent = fmt(t);
      fill.style.width = `${(100 * t / v.duration).toFixed(3)}%`; subs.forEach(f => f(t));
    };
    const state = () => {
      const p = video.paused; pp.innerHTML = p ? ICON.play : ICON.pause; pp.setAttribute("aria-label", p ? "Play" : "Pause");
      el.classList.toggle("playing", !p);
    };
    pp.onclick = toggle; big.onclick = toggle; video.onclick = toggle;
    range.oninput = () => { video.currentTime = +range.value; sync(); };
    el.querySelector(".pl-fs").onclick = () => (el.querySelector(".pl-frame").requestFullscreen || (() => {})).call(el.querySelector(".pl-frame"));
    video.addEventListener("timeupdate", sync); video.addEventListener("seeked", sync);
    video.addEventListener("play", () => { all.forEach(o => o !== video && o.pause()); state(); });
    video.addEventListener("pause", state); video.addEventListener("ended", state);
    el.addEventListener("keydown", e => {
      if (e.target === range) return;
      if (e.key === " " || e.key === "k") { e.preventDefault(); toggle(); }
      if (e.key === "ArrowRight") video.currentTime = Math.min(v.duration, video.currentTime + 5);
      if (e.key === "ArrowLeft") video.currentTime = Math.max(0, video.currentTime - 5);
    });
    all.add(video);
    if (opt.start) video.addEventListener("loadedmetadata", () => { video.currentTime = opt.start; sync(); }, { once: true });
    return { el, video, seek(t, play = true) { video.currentTime = t; sync(); if (play) video.play(); }, onTime(f) { subs.push(f); } };
  };
  // shared player CSS (pages set the --pl-* variables)
  const css = `
  .pl{--pl-ink:#1b1a19;--pl-muted:#6f6a66;--pl-track:#e4e1dc;--pl-accent:#8D1C06;--pl-bg:transparent;--pl-frame:#fff;--pl-edge:#e4e1dc}
  .pl-frame{position:relative;aspect-ratio:16/9;background:var(--pl-frame);box-shadow:0 0 0 1px var(--pl-edge);overflow:hidden}
  .pl-frame:fullscreen{box-shadow:none}
  .pl-video{display:block;width:100%;height:100%;object-fit:contain;background:var(--pl-frame);cursor:pointer}
  .pl-big{position:absolute;left:50%;top:50%;width:68px;height:68px;margin:-34px 0 0 -34px;border-radius:50%;border:0;padding:0 0 0 5px;
    background:var(--pl-accent);color:#fff;display:grid;place-items:center;cursor:pointer;box-shadow:0 6px 18px rgba(0,0,0,.18),0 1px 3px rgba(0,0,0,.2);
    transition:transform .25s cubic-bezier(.2,.8,.2,1),opacity .25s}
  .pl-big svg{width:30px;height:30px}
  .pl-big:hover{transform:scale(1.06)}
  .pl.playing .pl-big{opacity:0;pointer-events:none;transform:scale(.9)}
  .pl-bar{display:flex;align-items:center;gap:.7rem;padding:.55rem 0 0;background:var(--pl-bg);color:var(--pl-ink)}
  .pl-btn{flex:none;width:32px;height:32px;border-radius:50%;border:1px solid var(--pl-track);background:transparent;color:var(--pl-ink);display:grid;place-items:center;cursor:pointer;padding:0}
  .pl-btn svg{width:16px;height:16px}
  .pl-btn:hover{border-color:var(--pl-accent);color:var(--pl-accent)}
  .pl-scrub{position:relative;flex:1;height:22px}
  .pl-scrub::before{content:"";position:absolute;left:0;right:0;top:10px;height:2px;background:var(--pl-track);border-radius:2px}
  .pl-fill{position:absolute;left:0;top:10px;height:2px;background:var(--pl-accent);border-radius:2px;width:0}
  .pl-ticks i{position:absolute;top:6px;width:1px;height:10px;background:var(--pl-muted)}
  .pl-scrub input{position:absolute;inset:0;width:100%;margin:0;opacity:0;cursor:pointer}
  .pl-scrub:has(input:focus-visible){outline:2px solid var(--pl-accent);outline-offset:2px;border-radius:3px}
  .pl-time{flex:none;font-size:.8rem;color:var(--pl-muted);font-variant-numeric:tabular-nums lining-nums;min-width:5.6em;text-align:right}
  @media (prefers-reduced-motion:reduce){.pl-big{transition:none}}`;
  document.head.insertAdjacentHTML("beforeend", `<style>${css}</style>`);
})();
