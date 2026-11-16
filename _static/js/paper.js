// Copy button of the landing's install line (code blocks elsewhere use sphinx-copybutton).
document.addEventListener("click", (e) => {
  const btn = e.target.closest("button.paper-copy");
  if (!btn) return;
  const text = btn.dataset.copy;
  const done = () => {
    const label = btn.textContent;
    btn.textContent = "Copied";
    setTimeout(() => (btn.textContent = label), 1400);
  };
  const select = () => {
    const code = btn.parentElement.querySelector("pre");
    if (!code) return;
    const range = document.createRange();
    range.selectNodeContents(code);
    const sel = getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  };
  try {
    navigator.clipboard.writeText(text).then(done, select);
  } catch (err) {
    select();
  }
});

// Outline: mark the section in view among the current page's entries (a small gold square, like a bookmark).
document.addEventListener("DOMContentLoaded", () => {
  const links = [...document.querySelectorAll('.paper-outline li.current a.reference.internal[href^="#"]')];
  if (!links.length || !("IntersectionObserver" in window)) return;
  const byId = new Map();
  links.forEach((a) => {
    const id = decodeURIComponent(a.getAttribute("href").slice(1));
    const el = id && document.getElementById(id);
    if (el) byId.set(el, a);
  });
  const mark = (a) => links.forEach((l) => l.classList.toggle("paper-here", l === a));
  const io = new IntersectionObserver(
    (entries) => entries.forEach((e) => { if (e.isIntersecting) mark(byId.get(e.target)); }),
    { rootMargin: "-12% 0px -78% 0px" }
  );
  byId.forEach((_, el) => io.observe(el));
});
