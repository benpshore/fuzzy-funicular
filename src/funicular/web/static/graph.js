/* Canvas viewer for the literature graph. Buttons and keyboard for pan/zoom (no dragging or
   pinching required); tap or arrow through nodes to read details. */
(function () {
  "use strict";
  var canvas = document.getElementById("graph-canvas");
  if (!canvas || !window.fetch) return;
  var ctx = canvas.getContext("2d");
  var detail = document.getElementById("graph-detail");
  var view = { scale: 1, x: 0, y: 0 };
  var data = { nodes: [], edges: [] };
  var selected = -1;
  var dark = matchMedia("(prefers-color-scheme: dark)").matches || document.documentElement.dataset.theme === "dark";
  if (document.documentElement.dataset.theme === "light") dark = false;
  var colors = { bg: dark ? "#1c1c1e" : "#ffffff", edge: dark ? "#48484a" : "#c7c7cc", sim: dark ? "#636366" : "#d1d1d6",
                 node: dark ? "#0a84ff" : "#007aff", ext: dark ? "#2c2c2e" : "#f2f2f7", ring: dark ? "#98989f" : "#8e8e93",
                 text: dark ? "#ffffff" : "#000000", sel: dark ? "#ffd60a" : "#ff9500" };
  function radius(n) { return 6 + Math.min(14, Math.sqrt(n.degree || 0) * 3); }
  function draw() {
    var w = canvas.width, h = canvas.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = colors.bg; ctx.fillRect(0, 0, w, h);
    ctx.setTransform(view.scale, 0, 0, view.scale, view.x, view.y);
    var byId = {}; data.nodes.forEach(function (n) { byId[n.id] = n; });
    data.edges.forEach(function (e) {
      var a = byId[e.source], b = byId[e.target]; if (!a || !b) return;
      ctx.strokeStyle = e.kind === "similar" ? colors.sim : colors.edge; ctx.lineWidth = e.kind === "similar" ? 1 : 1.5;
      ctx.setLineDash(e.kind === "similar" ? [4, 4] : []);
      ctx.beginPath(); ctx.moveTo(a.x, a.y * 0.7); ctx.lineTo(b.x, b.y * 0.7); ctx.stroke();
    });
    ctx.setLineDash([]);
    data.nodes.forEach(function (n, i) {
      var r = radius(n);
      ctx.beginPath(); ctx.arc(n.x, n.y * 0.7, r, 0, Math.PI * 2);
      ctx.fillStyle = n.in_library ? colors.node : colors.ext; ctx.fill();
      ctx.lineWidth = i === selected ? 4 : 1.5; ctx.strokeStyle = i === selected ? colors.sel : colors.ring; ctx.stroke();
      if (view.scale >= 0.9 || i === selected || n.in_library) {
        ctx.fillStyle = colors.text; ctx.font = (i === selected ? "bold " : "") + "12px -apple-system, Helvetica, Arial, sans-serif";
        ctx.fillText((n.title || n.id).slice(0, 38) + (n.title && n.title.length > 38 ? "…" : ""), n.x + r + 4, n.y * 0.7 + 4);
      }
    });
  }
  function describe(i) {
    selected = i; draw();
    var n = data.nodes[i]; if (!n) return;
    var links = data.edges.filter(function (e) { return e.source === n.id || e.target === n.id; }).length;
    detail.innerHTML = "";
    var strong = document.createElement("strong"); strong.textContent = n.title || n.id; detail.appendChild(strong);
    detail.appendChild(document.createTextNode(" · " + (n.year || "") + " · " + links + " links" + (n.cited_by != null ? " · cited by " + n.cited_by : "") + (n.in_library ? " · in library" : "")));
    if (n.doc_id) { var a = document.createElement("a"); a.href = "/doc/" + encodeURIComponent(n.doc_id); a.textContent = " Open"; a.className = "btn btn-quiet"; detail.appendChild(a); }
    else if (n.doi) { var d = document.createElement("a"); d.href = "https://doi.org/" + n.doi; d.rel = "noreferrer"; d.textContent = " doi"; d.className = "btn btn-quiet"; detail.appendChild(d); }
  }
  function hit(px, py) {
    var x = (px - view.x) / view.scale, y = (py - view.y) / view.scale;
    var best = -1, bestD = 1e9;
    data.nodes.forEach(function (n, i) { var d = Math.hypot(n.x - x, n.y * 0.7 - y); if (d < radius(n) + 14 && d < bestD) { best = i; bestD = d; } });
    return best;
  }
  canvas.addEventListener("click", function (e) {
    var r = canvas.getBoundingClientRect();
    var i = hit((e.clientX - r.left) * canvas.width / r.width, (e.clientY - r.top) * canvas.height / r.height);
    if (i >= 0) describe(i);
  });
  canvas.addEventListener("keydown", function (e) {
    var step = 40;
    if (e.key === "ArrowRight" && !e.shiftKey) { selected = (selected + 1) % data.nodes.length; describe(selected); }
    else if (e.key === "ArrowLeft" && !e.shiftKey) { selected = (selected - 1 + data.nodes.length) % data.nodes.length; describe(selected); }
    else if (e.key === "+" || e.key === "=") { view.scale *= 1.2; draw(); }
    else if (e.key === "-") { view.scale /= 1.2; draw(); }
    else if (e.shiftKey && e.key === "ArrowLeft") { view.x += step; draw(); }
    else if (e.shiftKey && e.key === "ArrowRight") { view.x -= step; draw(); }
    else if (e.shiftKey && e.key === "ArrowUp") { view.y += step; draw(); }
    else if (e.shiftKey && e.key === "ArrowDown") { view.y -= step; draw(); }
    else return;
    e.preventDefault();
  });
  document.querySelectorAll("[data-g]").forEach(function (b) {
    b.addEventListener("click", function () {
      var k = b.dataset.g, step = 80;
      if (k === "zoom-in") view.scale *= 1.25; else if (k === "zoom-out") view.scale /= 1.25;
      else if (k === "left") view.x += step; else if (k === "right") view.x -= step;
      else if (k === "up") view.y += step; else if (k === "down") view.y -= step;
      else if (k === "reset") view = { scale: 1, x: 0, y: 0 };
      draw();
    });
  });
  fetch("/api/graph", { headers: { Accept: "application/json" }, credentials: "same-origin" })
    .then(function (r) { return r.json(); })
    .then(function (g) { data = g; draw(); });
})();
