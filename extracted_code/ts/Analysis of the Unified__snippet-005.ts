import { set, get } from 'idb-keyval';

export async function openDBPersistence() {
  // no-op for idb-keyval; ensure it is imported/available.
}

export async function saveSnapshot(key: string, payload: any) {
  try {
    await set(key, payload);
  } catch (err) {
    console.warn('persistence save failed', err);
  }
}

export async function loadSnapshot(key: string) {
  try {
    return await get(key);
  } catch (err) {
    console.warn('persistence load failed', err);
    return null;
  }
}
