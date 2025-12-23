function makeNeutral(emotions: EmotionName[]) {
const n: any = {};
for (const e of emotions) n[e] = 0;
