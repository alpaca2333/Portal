/**
 * Disk-based cache utility with stale-while-revalidate support.
 *
 * Behavior:
 *  - readCache(key)  → returns { data, isStale } or null if no cache exists on disk
 *  - writeCache(key, data) → persists data to disk with a timestamp
 *  - Cache files are stored under <cacheDir>/<key>.json
 *  - A cache entry is considered stale when its age exceeds CACHE_TTL_MS (1 hour)
 *
 * Stale-while-revalidate semantics are handled by the caller:
 *  - If null  → no cache; caller should fetch fresh data synchronously, then writeCache
 *  - If { data, isStale: false } → fresh cache; serve immediately
 *  - If { data, isStale: true  } → stale cache; serve immediately AND trigger a
 *                                   background refresh that calls writeCache when done
 */

import fs from 'fs';
import path from 'path';

// process.cwd() is always the project root when started via `npm run dev/start`
// This is more reliable than __dirname under tsx (which has no outDir).
export const CACHE_DIR = path.resolve(process.cwd(), 'cache/stock-data');
console.log(`[Cache] Cache directory: ${CACHE_DIR}`);

// 1 hour TTL
export const CACHE_TTL_MS = 60 * 60 * 1000;

export interface CacheEntry<T> {
    timestamp: number; // Unix ms
    data: T;
}

export interface CacheResult<T> {
    data: T;
    isStale: boolean;
}

/** Ensure the cache directory exists */
function ensureCacheDir(): void {
    if (!fs.existsSync(CACHE_DIR)) {
        fs.mkdirSync(CACHE_DIR, { recursive: true });
        console.log(`[Cache] Created cache directory: ${CACHE_DIR}`);
    }
}

/** Sanitise a cache key so it can be used as a filename */
function safeKey(key: string): string {
    return key.replace(/[^a-zA-Z0-9_\-]/g, '_');
}

function filePath(key: string): string {
    return path.join(CACHE_DIR, `${safeKey(key)}.json`);
}

/**
 * Read a cache entry from disk.
 * Returns null if no cache file exists.
 * Returns { data, isStale } otherwise.
 */
export function readCache<T>(key: string): CacheResult<T> | null {
    const fp = filePath(key);
    try {
        if (!fs.existsSync(fp)) return null;
        const raw = fs.readFileSync(fp, 'utf-8');
        const entry: CacheEntry<T> = JSON.parse(raw);
        const age = Date.now() - entry.timestamp;
        return { data: entry.data, isStale: age > CACHE_TTL_MS };
    } catch (err) {
        console.warn(`[Cache] Failed to read cache for key "${key}":`, err);
        return null;
    }
}

/**
 * Write data to disk cache with the current timestamp.
 */
export function writeCache<T>(key: string, data: T): void {
    ensureCacheDir();
    const fp = filePath(key);
    try {
        const entry: CacheEntry<T> = { timestamp: Date.now(), data };
        fs.writeFileSync(fp, JSON.stringify(entry), 'utf-8');
        console.log(`[Cache] Written cache for key "${key}"`);
    } catch (err) {
        console.warn(`[Cache] Failed to write cache for key "${key}":`, err);
    }
}
