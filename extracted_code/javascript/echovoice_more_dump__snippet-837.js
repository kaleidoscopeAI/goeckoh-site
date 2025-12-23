+ const WA = Array.from({ length: 8 }, () => Array.from({ length: 4 }, () => (Math.random() - 0.5) * 0.1)); // P x dE
+ const Winj = Array.from({ length: m }, () => Array.from({ length: WA.length }, () => Math.random() * 1e-3));
