// Applied before first paint to avoid a flash: system | light | dark (stored per device).
(function () {
  try {
    var t = localStorage.getItem("funicular.theme");
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
  } catch (e) { /* storage may be unavailable; system theme applies */ }
})();
