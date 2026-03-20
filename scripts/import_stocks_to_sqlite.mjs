#!/usr/bin/env node
/**
 * import_stocks_to_sqlite.mjs
 *
 * One-time script: imports all_stocks_daily.csv into a SQLite database.
 * Usage:
 *   node scripts/import_stocks_to_sqlite.mjs
 *
 * Output: data/quant/processed/stocks.db
 */

import Database from 'better-sqlite3';
import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, '..');
const CSV_DIR  = path.join(PROJECT_ROOT, 'data/quant/processed');
const DB_FILE  = path.join(PROJECT_ROOT, 'data/quant/processed/stocks.db');

// Auto-detect chunk files (all_stocks_daily_chunk*.csv)
const CSV_FILES = fs.readdirSync(CSV_DIR)
  .filter(f => /^all_stocks_daily_chunk\d+\.csv$/.test(f))
  .sort((a, b) => {
    const numA = parseInt(a.match(/chunk(\d+)/)[1]);
    const numB = parseInt(b.match(/chunk(\d+)/)[1]);
    return numA - numB;
  })
  .map(f => path.join(CSV_DIR, f));

// ── Sanity check ──────────────────────────────────────────────────────────────
if (CSV_FILES.length === 0) {
  console.error(`No chunk files found matching all_stocks_daily_chunk*.csv in ${CSV_DIR}`);
  process.exit(1);
}
console.log(`Found ${CSV_FILES.length} chunk file(s):`);
CSV_FILES.forEach(f => console.log(`  - ${path.basename(f)}`));

// Remove existing DB so we always start fresh
if (fs.existsSync(DB_FILE)) {
  console.log(`Removing existing DB: ${DB_FILE}`);
  fs.unlinkSync(DB_FILE);
}

// ── Open DB & create schema ───────────────────────────────────────────────────
const db = new Database(DB_FILE);

// WAL mode for faster bulk inserts
db.pragma('journal_mode = WAL');
db.pragma('synchronous = NORMAL');
db.pragma('cache_size = -65536');   // 64 MB page cache
db.pragma('temp_store = MEMORY');

db.exec(`
  CREATE TABLE IF NOT EXISTS kline (
    code             TEXT    NOT NULL,
    date             TEXT    NOT NULL,
    open             REAL,
    high             REAL,
    low              REAL,
    close            REAL,
    volume           REAL,
    factor           REAL,
    pb               REAL,
    pe_ttm           REAL,
    free_market_cap  REAL,
    industry_code    TEXT,
    industry_name    TEXT,
    PRIMARY KEY (code, date)
  );

  CREATE TABLE IF NOT EXISTS stock_meta (
    code          TEXT PRIMARY KEY,
    industry_code TEXT,
    industry_name TEXT
  );
`);

// ── Prepared statements ───────────────────────────────────────────────────────
const insertKline = db.prepare(`
  INSERT OR REPLACE INTO kline
    (code, date, open, high, low, close, volume, factor, pb, pe_ttm, free_market_cap, industry_code, industry_name)
  VALUES
    (@code, @date, @open, @high, @low, @close, @volume, @factor, @pb, @pe_ttm, @free_market_cap, @industry_code, @industry_name)
`);

const insertMeta = db.prepare(`
  INSERT OR REPLACE INTO stock_meta (code, industry_code, industry_name)
  VALUES (@code, @industry_code, @industry_name)
`);

// Wrap in a transaction for bulk performance
const BATCH_SIZE = 50_000;
let batch = [];
let totalRows = 0;
let colIndex = {};
const startTime = Date.now();

function flushBatch() {
  const insertMany = db.transaction((rows) => {
    for (const row of rows) {
      insertKline.run(row);
      // Update meta if industry info present
      if (row.industry_code || row.industry_name) {
        insertMeta.run({
          code: row.code,
          industry_code: row.industry_code,
          industry_name: row.industry_name,
        });
      }
    }
  });
  insertMany(batch);
  totalRows += batch.length;
  batch = [];
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  process.stdout.write(`\r  Imported ${totalRows.toLocaleString()} rows (${elapsed}s)...`);
}

// ── Process chunk files sequentially ──────────────────────────────────────────
console.log(`\nOutput: ${DB_FILE}\n`);

let headerParsed = false;

async function processChunk(csvFile) {
  return new Promise((resolve, reject) => {
    console.log(`\n📂 Processing: ${path.basename(csvFile)}`);

    const rl = readline.createInterface({
      input: fs.createReadStream(csvFile, { encoding: 'utf8' }),
      crlfDelay: Infinity,
    });

    let isFirstLine = true;

    rl.on('line', (line) => {
      if (!line) return;

      // Every chunk has a header row; parse it from the first chunk,
      // skip it for subsequent chunks.
      if (isFirstLine) {
        isFirstLine = false;
        if (!headerParsed) {
          line.split(',').forEach((col, i) => { colIndex[col.trim()] = i; });
          headerParsed = true;
        }
        return;
      }

      const cols = line.split(',');
      const code = cols[colIndex['code']]?.trim().toUpperCase();
      const date = cols[colIndex['date']]?.trim();
      if (!code || !date) return;

      const parseNum = (key) => {
        const v = cols[colIndex[key]]?.trim();
        return v === '' || v == null ? null : parseFloat(v);
      };
      const parseStr = (key) => {
        const v = cols[colIndex[key]]?.trim();
        return v === '' || v == null ? null : v;
      };

      batch.push({
        code,
        date,
        open:            parseNum('open'),
        high:            parseNum('high'),
        low:             parseNum('low'),
        close:           parseNum('close'),
        volume:          parseNum('volume'),
        factor:          parseNum('factor'),
        pb:              parseNum('pb'),
        pe_ttm:          parseNum('pe_ttm'),
        free_market_cap: parseNum('free_market_cap'),
        industry_code:   parseStr('industry_code'),
        industry_name:   parseStr('industry_name'),
      });

      if (batch.length >= BATCH_SIZE) {
        flushBatch();
      }
    });

    rl.on('close', () => {
      if (batch.length > 0) flushBatch();
      resolve();
    });

    rl.on('error', (err) => {
      reject(err);
    });
  });
}

// Main: process all chunks then build indexes
(async () => {
  try {
    for (const csvFile of CSV_FILES) {
      await processChunk(csvFile);
    }

    // Build indexes after all bulk inserts (much faster than building during insert)
    console.log('\n\nBuilding indexes...');
    db.exec(`
      CREATE INDEX IF NOT EXISTS idx_kline_code      ON kline(code);
      CREATE INDEX IF NOT EXISTS idx_kline_code_date ON kline(code, date);
      CREATE INDEX IF NOT EXISTS idx_kline_date      ON kline(date);
    `);

    // Optimize
    db.exec('ANALYZE;');
    db.pragma('wal_checkpoint(TRUNCATE)');
    db.close();

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const dbSize = (fs.statSync(DB_FILE).size / 1024 / 1024).toFixed(1);
    console.log(`\n✅ Done! ${totalRows.toLocaleString()} rows imported in ${elapsed}s`);
    console.log(`   DB size: ${dbSize} MB → ${DB_FILE}`);
  } catch (err) {
    console.error('\nError processing CSV:', err);
    db.close();
    process.exit(1);
  }
})();
