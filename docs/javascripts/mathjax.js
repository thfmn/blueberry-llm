window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Re-typeset on navigation (Material for MkDocs' instant loading)
document$.subscribe(() => {
  if (window.MathJax && window.MathJax.typesetPromise) {
    MathJax.startup?.output?.clearCache?.();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  }
});

