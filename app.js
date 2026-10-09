/**
 * Tracks reading progress and switches the header once the hero scrolls away.
 */
function initPageChrome() {
  const header = document.getElementById("site-header");
  const progress = document.querySelector(".read-progress");
  const hero = document.querySelector(".hero");
  const links = [...document.querySelectorAll(".header-nav a")];
  const sections = links
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  /**
   * Updates the progress bar, header theme, and current nav item.
   */
  function onScroll() {
    const scrolled = window.scrollY;
    const max = document.documentElement.scrollHeight - window.innerHeight;
    if (progress) {
      progress.style.width = `${max > 0 ? (scrolled / max) * 100 : 0}%`;
    }
    const heroBottom = hero ? hero.offsetHeight - 80 : 0;
    header?.classList.toggle("is-light", scrolled > heroBottom);

    const marker = scrolled + 140;
    let current = sections[0];
    sections.forEach((section) => {
      if (section.offsetTop <= marker) current = section;
    });
    links.forEach((link) => {
      const active = current && link.getAttribute("href") === `#${current.id}`;
      if (active) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    });
  }

  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });
}

document.addEventListener("DOMContentLoaded", initPageChrome);
