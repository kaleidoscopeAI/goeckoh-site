export default function EmotionalDial({ emotions, value, onChange, onApply, smoothingMs = 200 }: Props) {
const size = 400;
const cx = size / 2;
const cy = size / 2;
const radius = size * 0.35;
