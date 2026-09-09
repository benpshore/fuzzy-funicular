/* Funicular front end: theme toggle, live progress over SSE, upload without page reload.
   Everything degrades: forms post normally when JS is off. No timers dismiss anything. */
(function () {
  "use strict";
  var csrfMeta = document.querySelector('meta[name="csrf"]');
  var CSRF = csrfMeta ? csrfMeta.content : "";

  // ---------- theme ----------
  var choices = document.querySelectorAll("[data-theme-choice]");
  function applyTheme(t) {
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
    choices.forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.themeChoice === t)); });
    try { localStorage.setItem("funicular.theme", t); } catch (e) {}
  }
  var stored = "system";
  try { stored = localStorage.getItem("funicular.theme") || "system"; } catch (e) {}
  choices.forEach(function (b) {
    b.setAttribute("aria-pressed", String(b.dataset.themeChoice === stored));
    b.addEventListener("click", function () { applyTheme(b.dataset.themeChoice); });
  });

  // ---------- progress rendering ----------
  function el(id) { return document.querySelector('[data-doc-id="' + CSS.escape(id) + '"]'); }
  function setStatus(node, ev) {
    var badge = node.querySelector('[data-role="status"]');
    var bar = node.querySelector(".progress");
    var stage = node.querySelector(".row-stage");
    node.className = node.className.replace(/\bstatus-\w+/g, "").trim() + " status-" + ev.status;
    if (ev.status === "done") {
      if (bar) bar.hidden = true;
      if (stage) stage.textContent = "";
      if (badge) {
        if (ev.needs_ocr) { badge.textContent = "Needs OCR"; badge.className = "badge badge-warn"; }
        else if (ev.warnings && ev.warnings.length) {
          badge.textContent = ev.warnings.length + (ev.warnings.length === 1 ? " note" : " notes");
          badge.className = "badge badge-warn";
        } else { badge.textContent = "Ready"; badge.className = "badge badge-ok"; }
      }
      refreshRow(ev.id);
    } else if (ev.status === "failed") {
      if (bar) bar.hidden = true;
      if (stage) stage.textContent = ev.error ? "Failed: " + ev.error : "Failed";
      if (badge) { badge.textContent = "Failed"; badge.className = "badge badge-bad"; }
    } else {
      if (bar) {
        bar.hidden = false;
        bar.setAttribute("aria-valuenow", String(Math.round(ev.progress || 0)));
        var inner = bar.querySelector(".bar");
        if (inner) inner.style.width = (ev.progress || 0) + "%";
      }
      if (stage) stage.textContent = ev.label || ev.stage || "";
      if (badge) { badge.textContent = ev.label || "Working"; badge.className = "badge"; }
    }
  }
  function refreshRow(id) {
    fetch("/api/docs/" + encodeURIComponent(id), { headers: { Accept: "application/json" }, credentials: "same-origin" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        var node = el(id); if (!node) return;
        var title = node.querySelector(".row-title"); if (title) title.textContent = d.title;
        var sub = node.querySelector(".row-sub");
        if (sub) sub.textContent = d.kind.toUpperCase() + " · " + humanSize(d.size) + (d.pages ? " · " + d.pages + " p" : "") + " · just now";
        // On the document page, reload once so the text viewer appears (no auto-refresh loops).
        if (document.body.querySelector(".status-card") && !node.dataset.reloaded) {
          node.dataset.reloaded = "1";
          window.location.reload();
        }
      });
  }
  function humanSize(n) {
    var u = ["B", "KB", "MB", "GB"], i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return (i ? n.toFixed(1) : n) + " " + u[i];
  }

  // ---------- batches (folder / archive imports) ----------
  function batchEl(id) { return document.querySelector('[data-batch-id="' + CSS.escape(id) + '"]'); }
  function renderBatch(ev) {
    var section = document.getElementById("batches");
    var list = document.getElementById("batch-list");
    var tpl = document.getElementById("batch-row-template");
    if (!section || !list || !tpl) return;
    var node = batchEl(ev.id);
    if (!node) {
      node = tpl.content.firstElementChild.cloneNode(true);
      node.dataset.batchId = ev.id;
      node.querySelector(".row-title").textContent = (ev.kind === "archive" ? "\uD83D\uDDDC\uFE0F " : "\uD83D\uDCC1 ") + (ev.source || "");
      list.insertBefore(node, list.firstChild);
    }
    section.hidden = false;
    node.className = "batch-item status-" + ev.status;
    var badge = node.querySelector('[data-role="status"]');
    var bar = node.querySelector(".progress");
    var detail = node.querySelector('[data-role="detail"]');
    var pct = ev.total ? Math.round(100 * ev.done / ev.total) : 0;
    if (badge) { badge.textContent = ev.status.charAt(0).toUpperCase() + ev.status.slice(1); badge.className = "badge" + (ev.status === "done" ? " badge-ok" : ev.status === "failed" ? " badge-bad" : ""); }
    if (bar) { bar.hidden = ev.status === "done" || ev.status === "failed"; bar.setAttribute("aria-valuenow", String(pct)); bar.querySelector(".bar").style.width = pct + "%"; }
    if (detail) {
      var parts = [ev.done + "/" + ev.total];
      if (ev.imported !== undefined) parts.push(ev.imported + " imported");
      if (ev.skipped !== undefined) parts.push(ev.skipped + " skipped");
      if (ev.errors) parts.push(ev.errors + " error(s)");
      if (ev.error) parts.push(ev.error);
      detail.textContent = parts.join(" \u00B7 ");
    }
    if (ev.status === "done" && list && !node.dataset.refreshed) {
      node.dataset.refreshed = "1";
      fetch("/api/docs", { headers: { Accept: "application/json" }, credentials: "same-origin" })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (j) { if (!j) return; j.docs.slice(0, 50).reverse().forEach(function (d) { if (!el(d.id)) addRow(d); }); });
    }
  }

  // ---------- SSE ----------
  if (window.EventSource && (document.getElementById("doc-list") || document.querySelector("[data-doc-id]") || document.getElementById("batches"))) {
    var es = new EventSource("/events", { withCredentials: true });
    var handle = function (e) {
      var ev; try { ev = JSON.parse(e.data); } catch (err) { return; }
      if (ev.type === "snapshot") {
        (ev.docs || []).forEach(function (d) { var n = el(d.id); if (n) setStatus(n, d); });
        (ev.batches || []).forEach(function (b) { renderBatch({ id: b.id, kind: b.kind, status: b.status, source: b.source.split("/").pop(), done: b.done, total: b.total, imported: b.imported, skipped: b.skipped }); });
        return;
      }
      if (ev.type === "batch") { renderBatch(ev); return; }
      var node = el(ev.id); if (node) setStatus(node, ev);
    };
    ["progress", "done", "failed", "snapshot", "batch"].forEach(function (t) { es.addEventListener(t, handle); });
  }

  // ---------- upload ----------
  var form = document.getElementById("upload-form");
  var list = document.getElementById("doc-list");
  var tpl = document.getElementById("doc-row-template");
  var status = document.getElementById("upload-status");
  var section = form ? form.closest(".upload") : null;
  function addRow(d) {
    if (!list || !tpl) return;
    var empty = list.querySelector(".empty"); if (empty) empty.remove();
    var node = tpl.content.firstElementChild.cloneNode(true);
    node.dataset.docId = d.id;
    node.querySelector(".row-link").href = "/doc/" + encodeURIComponent(d.id);
    var pick = node.querySelector(".pick input"); if (pick) { pick.value = d.id; pick.setAttribute("aria-label", "Select " + d.title); }
    node.querySelector(".row-title").textContent = d.title;
    node.querySelector(".row-sub").textContent = d.kind.toUpperCase() + " · " + humanSize(d.size) + " · just now";
    node.querySelector(".progress").setAttribute("aria-label", "Processing " + d.title);
    setStatus(node, { status: "queued", label: "Queued", progress: 0 });
    list.insertBefore(node, list.firstChild);
  }
  function send(files) {
    if (!files || !files.length) return;
    var fd = new FormData();
    fd.append("csrf", CSRF);
    var ocr = form.querySelector('input[name="ocr"]:checked');
    fd.append("ocr", ocr ? ocr.value : "off");
    for (var i = 0; i < files.length; i++) fd.append("files", files[i], files[i].name);
    var btn = document.getElementById("upload-submit");
    if (btn) btn.disabled = true;
    status.textContent = "Uploading " + files.length + (files.length === 1 ? " file…" : " files…");
    fetch("/upload", { method: "POST", body: fd, credentials: "same-origin",
      headers: { Accept: "application/json", "X-CSRF-Token": CSRF } })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (res) {
        var j = res.j || {};
        (j.created || []).forEach(addRow);
        (j.batches || []).forEach(function (b) { renderBatch({ id: b.id, kind: "archive", status: "queued", source: b.name, done: 0, total: 0 }); });
        var msg = [];
        if (j.created && j.created.length) msg.push(j.created.length + " added");
        if (j.batches && j.batches.length) msg.push(j.batches.length + " archive(s) unpacking");
        (j.errors || []).forEach(function (e) { msg.push(e.name + ": " + e.error); });
        if (!res.ok && j.error) msg.push(j.error);
        status.textContent = msg.join(" · ") || "Nothing uploaded";
        form.reset();
      })
      .catch(function () { status.textContent = "Upload failed. Check the connection and try again."; })
      .then(function () { if (btn) btn.disabled = false; });
  }
  if (form && window.fetch && window.FormData) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      send(document.getElementById("files").files);
    });
    // Drag and drop is a convenience only; the button path is the primary one.
    if (section) {
      ["dragenter", "dragover"].forEach(function (t) { section.addEventListener(t, function (e) { e.preventDefault(); section.classList.add("dragover"); }); });
      ["dragleave", "drop"].forEach(function (t) { section.addEventListener(t, function (e) { e.preventDefault(); section.classList.remove("dragover"); }); });
      section.addEventListener("drop", function (e) { if (e.dataTransfer) send(e.dataTransfer.files); });
    }
  }

  // ---------- select all for export ----------
  var selectAll = document.getElementById("select-all");
  if (selectAll) selectAll.addEventListener("change", function () {
    document.querySelectorAll('#doc-list input[name="ids"]').forEach(function (b) { b.checked = selectAll.checked; });
  });

  // ---------- destructive forms: require the checkbox, never a timed dialog ----------
  document.querySelectorAll("form[data-confirm]").forEach(function (f) {
    f.addEventListener("submit", function (e) {
      var box = f.querySelector('input[name="confirm"]');
      if (box && !box.checked) { e.preventDefault(); box.focus(); }
    });
  });
})();
