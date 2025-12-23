const mean = data[0].map((_, i) => data.reduce((sum, row) => sum + row[i], 0) / data.length);
return mean.map(() => mean.map(() => 0)); // Placeholder, implement full
