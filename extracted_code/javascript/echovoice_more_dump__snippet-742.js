const map = new Map(items.map(i => [i.id, i]));
for (const item of fromDb) if (!map.has(item.id)) map.set(item.id, item);
