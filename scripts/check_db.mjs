#!/usr/bin/env node
/**
 * check_db.mjs  —  Quick SQLite health check
 * Usage: node scripts/check_db.mjs
 */
import Database from 'better-sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_PATH = path.resolve(__dirname, '..', 'data/quant/processed/stocks.db');

console.log(`\n📂 DB path: ${DB_PATH}\n`);

const db = new Database(DB_PATH, { readonly: true });

// 1. List all tables
const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").all();
console.log('═══ Tables ═══');
tables.forEach(t => console.log(`  • ${t.name}`));

// 2. kline table schema (all columns)
console.log('\n═══ kline columns ═══');
const cols = db.pragma('table_info(kline)');
cols.forEach(c => console.log(`  ${c.cid}  ${c.name.padEnd(20)} ${c.type.padEnd(8)} ${c.notnull ? 'NOT NULL' : 'nullable'}${c.pk ? '  PK' : ''}`));

// 3. Row count
const totalRows = db.prepare('SELECT COUNT(*) as cnt FROM kline').get().cnt;
const totalStocks = db.prepare('SELECT COUNT(DISTINCT code) as cnt FROM kline').get().cnt;
console.log(`\n═══ Stats ═══`);
console.log(`  Total rows:   ${totalRows.toLocaleString()}`);
console.log(`  Total stocks: ${totalStocks.toLocaleString()}`);

// 4. Date range
const dateRange = db.prepare('SELECT MIN(date) as min_date, MAX(date) as max_date FROM kline').get();
console.log(`  Date range:   ${dateRange.min_date} ~ ${dateRange.max_date}`);

// 5. Check if roe_ttm column exists
const hasROE = cols.some(c => c.name === 'roe_ttm');
console.log(`  roe_ttm col:  ${hasROE ? '✅ EXISTS' : '❌ MISSING'}`);
if (hasROE) {
  const roeCount = db.prepare('SELECT COUNT(*) as cnt FROM kline WHERE roe_ttm IS NOT NULL').get().cnt;
  console.log(`  roe_ttm fill: ${roeCount.toLocaleString()} / ${totalRows.toLocaleString()} (${(roeCount/totalRows*100).toFixed(1)}%)`);
}

// 6. Sample: 600519 (Maotai) — try both SH600519 and sh600519
console.log('\n═══ Sample: 600519 (茅台) ═══');
const codesToTry = ['SH600519', 'sh600519', '600519.SH', '600519'];

for (const code of codesToTry) {
  const cnt = db.prepare('SELECT COUNT(*) as cnt FROM kline WHERE code = ?').get(code).cnt;
  if (cnt > 0) {
    console.log(`  Found with code = '${code}' (${cnt} rows)`);
    const rows = db.prepare('SELECT * FROM kline WHERE code = ? ORDER BY date DESC LIMIT 2').all(code);
    rows.forEach(r => {
      console.log(`\n  📅 ${r.date}`);
      for (const [k, v] of Object.entries(r)) {
        if (k !== 'date') console.log(`     ${k.padEnd(20)} = ${v}`);
      }
    });
    break;
  } else {
    console.log(`  ❌ code='${code}' → 0 rows`);
  }
}

// 7. Show a few distinct code formats for reference
console.log('\n═══ Sample codes (first 10) ═══');
const sampleCodes = db.prepare('SELECT DISTINCT code FROM kline ORDER BY code LIMIT 10').all();
sampleCodes.forEach(r => console.log(`  ${r.code}`));

db.close();
console.log('\n✅ Done.\n');
