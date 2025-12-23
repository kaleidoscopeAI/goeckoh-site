const sub = m.map(row => row.slice(0, i).concat(row.slice(i + 1)));
const sign = i % 2 === 0 ? 1 : -1;
sum += sign * m[0][i] * det(sub.slice(1));
