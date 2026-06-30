/* gate.js — soft early-access gate
 * Include in any tool page to require early-access signup.
 * Does NOT block the early-access page itself or the demo page.
 */
(function () {
  'use strict';

  const DEMO_PAGES  = ['demo.html', 'index.html', 'download.html', ''];
  const path        = location.pathname.split('/').pop() || '';

  if (DEMO_PAGES.includes(path)) return;

  const email = localStorage.getItem('gk_access');
  if (!email) {
    sessionStorage.setItem('gk_return', location.href);
    location.replace('download.html');
  }
})();
