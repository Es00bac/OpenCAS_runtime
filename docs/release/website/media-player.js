(function () {
  const VIDEO_SRC = "https://github.com/Es00bac/OpenCAS_runtime/releases/download/media-2026-04-14/OpenCAS_Animated_Final.mp4";
  const VIDEO_TYPE = "video/mp4";

  function createPlayer() {
    if (document.getElementById("ocas-video-toggle")) return;

    const toggle = document.createElement("button");
    toggle.id = "ocas-video-toggle";
    toggle.innerHTML = `
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>
      <span>Open the signal reel</span>
    `;

    const panel = document.createElement("div");
    panel.id = "ocas-video-panel";
    panel.innerHTML = `
      <div id="ocas-video-header">
        <div id="ocas-video-title">OpenCAS: Signal Reel</div>
        <div id="ocas-video-controls">
          <button id="ocas-video-expand" title="Expand">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5h6v2H7v4H5V5zm8 0h6v6h-2V7h-4V5zM5 13h2v4h4v2H5v-6zm12 0h2v6h-6v-2h4v-4z"/></svg>
          </button>
          <button id="ocas-video-close" title="Close">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18.3 5.71L12 12l6.3 6.29-1.41 1.42L10.59 13.4l-6.3 6.29L3.88 18.3 10.17 12 3.88 5.71 5.29 4.29 11.59 10.6 17.88 4.29z"/></svg>
          </button>
        </div>
      </div>
      <div id="ocas-video-wrapper">
        <video controls playsinline preload="metadata" crossorigin="anonymous">
          <source src="${VIDEO_SRC}" type="${VIDEO_TYPE}">
        </video>
      </div>
    `;

    document.body.appendChild(toggle);
    document.body.appendChild(panel);

    const video = panel.querySelector("video");
    const expandBtn = document.getElementById("ocas-video-expand");
    const closeBtn = document.getElementById("ocas-video-close");
    let expanded = false;

    function openPanel() {
      panel.classList.add("ocas-open");
      video.setAttribute("controls", "");
    }

    function closePanel() {
      panel.classList.remove("ocas-open", "ocas-expanded");
      expanded = false;
      video.pause();
    }

    function toggleExpand() {
      expanded = !expanded;
      panel.classList.toggle("ocas-expanded", expanded);
      expandBtn.title = expanded ? "Collapse" : "Expand";
      expandBtn.innerHTML = expanded
        ? `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 9H5V5h4v4zm6 0V5h4v4h-4zm-6 6H5v-4h4v4zm6 0v-4h4v4h-4z"/></svg>`
        : `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5h6v2H7v4H5V5zm8 0h6v6h-2V7h-4V5zM5 13h2v4h4v2H5v-6zm12 0h2v6h-6v-2h4v-4z"/></svg>`;
    }

    toggle.addEventListener("click", openPanel);
    closeBtn.addEventListener("click", closePanel);
    expandBtn.addEventListener("click", toggleExpand);

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && expanded) {
        toggleExpand();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", createPlayer);
  } else {
    createPlayer();
  }
})();
