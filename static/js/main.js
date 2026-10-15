(function () {
  'use strict';

  // ---------- Table of contents: toggle + scroll spy ----------
  var toc = document.getElementById('toc');
  var toggle = toc && toc.querySelector('.toc-toggle');
  var links = toc ? Array.prototype.slice.call(toc.querySelectorAll('.toc-panel a')) : [];

  function setOpen(open) {
    if (!toc) return;
    toc.classList.toggle('open', open);
    toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  if (toggle) {
    toggle.addEventListener('click', function (e) {
      e.stopPropagation();
      setOpen(!toc.classList.contains('open'));
    });
    document.addEventListener('click', function (e) {
      if (toc.classList.contains('open') && !toc.contains(e.target)) setOpen(false);
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') setOpen(false);
    });
    links.forEach(function (a) {
      a.addEventListener('click', function () {
        if (window.innerWidth < 1440) setOpen(false);
      });
    });
  }

  var sections = links
    .map(function (a) { return document.querySelector(a.getAttribute('href')); })
    .filter(Boolean);

  function setActive(id) {
    links.forEach(function (a) {
      a.classList.toggle('active', a.getAttribute('href') === '#' + id);
    });
  }

  if ('IntersectionObserver' in window && sections.length) {
    var visible = {};
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { visible[en.target.id] = en.isIntersecting; });
      for (var i = 0; i < sections.length; i++) {
        if (visible[sections[i].id]) { setActive(sections[i].id); break; }
      }
    }, { rootMargin: '-35% 0px -55% 0px' });
    sections.forEach(function (s) { io.observe(s); });
  }

  // ---------- Back to top ----------
  var toTop = document.getElementById('toTop');
  if (toTop) {
    var onScroll = function () { toTop.classList.toggle('show', window.scrollY > 700); };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
    toTop.addEventListener('click', function () { window.scrollTo({ top: 0 }); });
  }

  // ---------- Copy BibTeX ----------
  var copyBtn = document.getElementById('copyBib');
  var bibText = document.getElementById('bibText');
  if (copyBtn && bibText) {
    copyBtn.addEventListener('click', function () {
      var label = copyBtn.querySelector('span');
      var done = function () {
        copyBtn.classList.add('done');
        label.textContent = 'Copied';
        setTimeout(function () { copyBtn.classList.remove('done'); label.textContent = 'Copy'; }, 1800);
      };
      // If the clipboard is unavailable, leave the BibTeX selected so it can be copied by hand.
      var fallback = function () {
        var r = document.createRange();
        r.selectNodeContents(bibText);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(r);
        try { if (document.execCommand('copy')) done(); } catch (err) { /* text stays selected */ }
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(bibText.textContent).then(done, fallback);
      } else {
        fallback();
      }
    });
  }

  // ---------- Placeholder links (data-todo) ----------
  // Buttons without a real URL yet do nothing instead of jumping to the top.
  Array.prototype.forEach.call(document.querySelectorAll('a[data-todo="true"]'), function (a) {
    a.setAttribute('title', 'Coming soon');
    a.addEventListener('click', function (e) { e.preventDefault(); });
  });
})();
