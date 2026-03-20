#!/usr/bin/env node
/**
 * import_roe.mjs
 * ==============
 * Merge ROE data into:
 *   1. all_stocks_daily_chunk{0,1,2}.csv  (add roe_ttm column)
 *   2. stocks.db kline table              (add roe_ttm column)
 *
 * ROE source: /root/qlib_data/roe/{TICKER}.{EXCHANGE}.csv
 *   - Quarterly, keyed by announcement date
 *   - Fields: date, end_date, symbol, roe, roe_ttm, roe_deducted
 *   - symbol format: sz000001 (lowercase)
 *
 * Strategy: use announcement date as point-in-time marker,
 * forward-fill roe_ttm to each trading day (no look-ahead bias).
 *
 * Usage:
 *   node scripts/import_roe.mjs
 */

import Database from 'better-sqlite3';
import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, '..');

const ROE_DIR = 'data/quant/processed/roe';
const CHUNK_DIR = path.join(PROJECT_ROOT, 'data/quant/processed');
const DB_PATH = path.join(CHUNK_DIR, 'stocks.db');

// ─── Step 1: Load all ROE data into memory ───────────────────────────────────

/**
 * Load all ROE CSVs and return a Map<code_upper, Array<{annDate, roeTtm}>>
 * sorted by annDate ascending.
 */
async function loadAllROE() {
  const files = fs.readdirSync(ROE_DIR).filter(f => f.endsWith('.csv')).sort();
  console.log(`[roe] Found ${files.length} ROE files`);

  // Map: uppercase code -> [{annDate: 'YYYY-MM-DD', roeTtm: number}]
  const roeMap = new Map();
  let totalObs = 0;

  for (const file of files) {
    const fpath = path.join(ROE_DIR, file);
    const lines = fs.readFileSync(fpath, 'utf8').split('\n');
    if (lines.length < 2) continue;

    // Parse header
    const header = lines[0].trim().split(',');
    const idx = {};
    header.forEach((col, i) => { idx[col.trim()] = i; });
    if (!('roe_ttm' in idx) || !('date' in idx) || !('symbol' in idx)) continue;

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      const cols = line.split(',');
      const annDate = cols[idx['date']]?.trim();
      const symbol = cols[idx['symbol']]?.trim();
      const roeTtmStr = cols[idx['roe_ttm']]?.trim();

      if (!annDate || !symbol || !roeTtmStr || roeTtmStr === '') continue;
      const roeTtm = parseFloat(roeTtmStr);
      if (isNaN(roeTtm)) continue;

      const codeUpper = symbol.toUpperCase();
      if (!roeMap.has(codeUpper)) roeMap.set(codeUpper, []);
      roeMap.get(codeUpper).push({ annDate, roeTtm });
      totalObs++;
    }
  }

  // Sort each stock's ROE by announcement date
  for (const [code, arr] of roeMap) {
    arr.sort((a, b) => a.annDate.localeCompare(b.annDate));
  }

  console.log(`[roe] Loaded ${totalObs.toLocaleString()} valid observations for ${roeMap.size} stocks`);
  return roeMap;
}

/**
 * Given a sorted array of {annDate, roeTtm} and a target date string,
 * find the most recent ROE announced on or before that date.
 * Uses binary search for efficiency.
 */
function lookupROE(roeArr, dateStr) {
  if (!roeArr || roeArr.length === 0) return null;

  let lo = 0, hi = roeArr.length - 1;
  let result = null;

  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (roeArr[mid].annDate <= dateStr) {
      result = roeArr[mid].roeTtm;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return result;
}

// ─── Step 2: Process chunk CSV files ─────────────────────────────────────────

async function processChunks(roeMap) {
  const chunkFiles = fs.readdirSync(CHUNK_DIR)
    .filter(f => f.startsWith('all_stocks_daily_chunk') && f.endsWith('.csv'))
    .sort()
    .map(f => path.join(CHUNK_DIR, f));

  console.log(`\n[chunks] Found ${chunkFiles.length} chunk files`);

  for (const chunkFile of chunkFiles) {
    console.log(`\n[chunk] Processing ${path.basename(chunkFile)} ...`);
    const t0 = Date.now();

    // Stream read → tmp file → rename (handles multi-GB files)
    const tmpFile = chunkFile + '.tmp';
    const writeStream = fs.createWriteStream(tmpFile, { encoding: 'utf8' });

    let headerParsed = false;
    let codeIdx = -1, dateIdx = -1, roeTtmIdx = -1;
    let filled = 0, total = 0;

    await new Promise((resolve, reject) => {
      const rl = readline.createInterface({
        input: fs.createReadStream(chunkFile, { encoding: 'utf8' }),
        crlfDelay: Infinity,
      });

      rl.on('line', (line) => {
        if (!line.trim()) return;

        if (!headerParsed) {
          headerParsed = true;
          const headers = line.trim().split(',');
          codeIdx = headers.indexOf('code');
          dateIdx = headers.indexOf('date');
          roeTtmIdx = headers.indexOf('roe_ttm');

          if (codeIdx === -1 || dateIdx === -1) {
            console.log('  Missing code/date columns, skipping');
            rl.close();
            return;
          }

          if (roeTtmIdx !== -1) {
            console.log('  roe_ttm column exists, will overwrite');
            writeStream.write(line + '\n');
          } else {
            console.log('  Adding roe_ttm column');
            writeStream.write(line + ',roe_ttm\n');
          }
          return;
        }

        const cols = line.split(',');
        const code = cols[codeIdx]?.trim();
        const date = cols[dateIdx]?.trim();

        if (!code || !date) {
          writeStream.write(line + (roeTtmIdx === -1 ? ',' : '') + '\n');
          return;
        }

        total++;
        const codeUpper = code.toUpperCase();
        const roeArr = roeMap.get(codeUpper);
        const roeTtm = lookupROE(roeArr, date);

        if (roeTtmIdx !== -1) {
          cols[roeTtmIdx] = roeTtm != null ? roeTtm.toString() : '';
          writeStream.write(cols.join(',') + '\n');
        } else {
          writeStream.write(line + ',' + (roeTtm != null ? roeTtm.toString() : '') + '\n');
        }

        if (roeTtm != null) filled++;

        if (total % 2_000_000 === 0) {
          const elapsed = ((Date.now() - t0) / 1000).toFixed(0);
          process.stdout.write(`\r  Processed ${total.toLocaleString()} rows (${elapsed}s)...`);
        }
      });

      rl.on('close', () => {
        writeStream.end(() => resolve());
      });
      rl.on('error', reject);
    });

    // Replace original with tmp
    fs.renameSync(tmpFile, chunkFile);

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const pct = total > 0 ? (filled / total * 100).toFixed(1) : '0.0';
    console.log(`\n  Saved. ROE coverage: ${filled.toLocaleString()}/${total.toLocaleString()} (${pct}%) in ${elapsed}s`);
  }
}

// ─── Step 3: Update SQLite ───────────────────────────────────────────────────

function updateSQLite(roeMap) {
  console.log(`\n[sqlite] Updating ${DB_PATH} ...`);
  const db = new Database(DB_PATH);
  db.pragma('journal_mode = WAL');
  db.pragma('synchronous = NORMAL');
  db.pragma('cache_size = -65536');

  // Check if roe_ttm column exists
  const cols = db.pragma('table_info(kline)').map(c => c.name);
  if (!cols.includes('roe_ttm')) {
    console.log('  Adding roe_ttm column ...');
    db.exec('ALTER TABLE kline ADD COLUMN roe_ttm REAL');
  } else {
    console.log('  roe_ttm column already exists, will update values');
  }

  const selectDates = db.prepare('SELECT date FROM kline WHERE code = ? ORDER BY date');
  const updateStmt = db.prepare('UPDATE kline SET roe_ttm = ? WHERE code = ? AND date = ?');

  const codes = Array.from(roeMap.keys()); // uppercase
  console.log(`  Updating ROE for ${codes.length} stocks ...`);

  const t0 = Date.now();
  let totalUpdated = 0;

  // Batch in transaction
  const BATCH = 500;
  for (let batchStart = 0; batchStart < codes.length; batchStart += BATCH) {
    const batchEnd = Math.min(batchStart + BATCH, codes.length);
    const batchCodes = codes.slice(batchStart, batchEnd);

    const txn = db.transaction(() => {
      for (const code of batchCodes) {
        const roeArr = roeMap.get(code);
        if (!roeArr || roeArr.length === 0) continue;

        const dates = selectDates.all(code);
        for (const row of dates) {
          const roeTtm = lookupROE(roeArr, row.date);
          if (roeTtm != null) {
            updateStmt.run(roeTtm, code, row.date);
            totalUpdated++;
          }
        }
      }
    });
    txn();

    const elapsed = ((Date.now() - t0) / 1000).toFixed(0);
    process.stdout.write(`\r  [sqlite] ${batchEnd}/${codes.length} stocks, ${totalUpdated.toLocaleString()} rows updated (${elapsed}s)`);
  }

  console.log('');

  // Verify
  const totalWithROE = db.prepare('SELECT COUNT(*) as cnt FROM kline WHERE roe_ttm IS NOT NULL').get().cnt;
  const totalRows = db.prepare('SELECT COUNT(*) as cnt FROM kline').get().cnt;
  const pct = totalRows > 0 ? (totalWithROE / totalRows * 100).toFixed(1) : '0.0';
  console.log(`  Verification: ${totalWithROE.toLocaleString()}/${totalRows.toLocaleString()} rows have ROE (${pct}%)`);

  const samples = db.prepare(
    'SELECT code, date, roe_ttm FROM kline WHERE roe_ttm IS NOT NULL ORDER BY date DESC LIMIT 5'
  ).all();
  console.log('  Sample rows:');
  for (const s of samples) {
    console.log(`    ${s.code} | ${s.date} | roe_ttm=${s.roe_ttm}`);
  }

  db.exec('ANALYZE;');
  db.pragma('wal_checkpoint(TRUNCATE)');
  db.close();
  console.log('[sqlite] Done.');
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  console.log('='.repeat(64));
  console.log('ROE Data Import Script (Node.js)');
  console.log('='.repeat(64));

  console.log('\n[1/3] Loading all ROE data ...');
  const roeMap = await loadAllROE();

  console.log('\n[2/3] Merging into chunk files ...');
  await processChunks(roeMap);

  console.log('\n[3/3] Updating SQLite database ...');
  updateSQLite(roeMap);

  console.log('\n✅ All done!');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
