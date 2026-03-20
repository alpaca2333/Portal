import { FastifyInstance } from 'fastify';
import { parse } from 'csv-parse';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import Database, { Database as DatabaseType } from 'better-sqlite3';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Data files are stored inside the portal project under data/quant/
const PROJECT_ROOT = path.resolve(__dirname, '../../');
const DB_FILE     = path.join(PROJECT_ROOT, 'data/quant/processed/stocks.db');
const BACKTEST_DIR = path.join(PROJECT_ROOT, 'data/quant/backtest');

// ─── Types ───────────────────────────────────────────────────────────────────

interface KlinePoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// ─── SQLite connection ────────────────────────────────────────────────────────

let db: DatabaseType | null = null;
let dbReady = false;
let stockCount = 0;

function openDb(log: (msg: string) => void): void {
  if (!fs.existsSync(DB_FILE)) {
    log(`[Quant DB] ⚠️  SQLite DB not found: ${DB_FILE}`);
    log(`[Quant DB]    Run: node scripts/import_stocks_to_sqlite.mjs`);
    return;
  }

  try {
    db = new Database(DB_FILE, { readonly: true });
    db.pragma('journal_mode = WAL');
    db.pragma('cache_size = -32768');  // 32 MB page cache
    db.pragma('temp_store = MEMORY');

    // Count distinct stocks
    const row = db.prepare('SELECT COUNT(DISTINCT code) AS cnt FROM kline').get() as { cnt: number };
    stockCount = row.cnt;
    dbReady = true;
    log(`[Quant DB] ✅ SQLite ready: ${stockCount.toLocaleString()} stocks — ${DB_FILE}`);
  } catch (err: any) {
    log(`[Quant DB] ❌ Failed to open SQLite: ${err.message}`);
  }
}

// ─── Query helpers ────────────────────────────────────────────────────────────

function getStockData(code: string, start?: string, end?: string): KlinePoint[] {
  if (!db) return [];

  const upperCode = code.toUpperCase();

  if (start && end) {
    return db.prepare(
      'SELECT date, open, high, low, close, volume FROM kline WHERE code = ? AND date >= ? AND date <= ? ORDER BY date'
    ).all(upperCode, start, end) as KlinePoint[];
  } else if (start) {
    return db.prepare(
      'SELECT date, open, high, low, close, volume FROM kline WHERE code = ? AND date >= ? ORDER BY date'
    ).all(upperCode, start) as KlinePoint[];
  } else if (end) {
    return db.prepare(
      'SELECT date, open, high, low, close, volume FROM kline WHERE code = ? AND date <= ? ORDER BY date'
    ).all(upperCode, end) as KlinePoint[];
  } else {
    return db.prepare(
      'SELECT date, open, high, low, close, volume FROM kline WHERE code = ? ORDER BY date'
    ).all(upperCode) as KlinePoint[];
  }
}

function searchCodes(query: string): string[] {
  if (!db) return [];
  const upperQuery = query.toUpperCase();
  const rows = db.prepare(
    'SELECT DISTINCT code FROM kline WHERE code LIKE ? ORDER BY code LIMIT 20'
  ).all(`%${upperQuery}%`) as { code: string }[];
  return rows.map((r) => r.code);
}

/**
 * Read a backtest CSV file and return its rows as objects.
 */
async function readBacktestCsv(filename: string): Promise<Record<string, string>[]> {
  const filePath = path.join(BACKTEST_DIR, filename);
  if (!fs.existsSync(filePath)) return [];

  return new Promise((resolve, reject) => {
    const rows: Record<string, string>[] = [];
    fs.createReadStream(filePath)
      .pipe(parse({ columns: true, trim: true }))
      .on('data', (row: Record<string, string>) => rows.push(row))
      .on('end', () => resolve(rows))
      .on('error', reject);
  });
}

// ─── Route plugin ─────────────────────────────────────────────────────────────

export default async function quantRoutes(fastify: FastifyInstance) {
  // Open SQLite on plugin init
  openDb((msg) => fastify.log.info(msg));

  // Health check
  fastify.get('/health', async () => ({ status: 'ok' }));

  // Loading status (kept for frontend compatibility — SQLite is instant)
  fastify.get('/status', async () => ({
    ready: dbReady,
    loadedRows: dbReady ? stockCount * 4000 : 0,  // approximate, just for progress bar
    stockCount,
  }));

  // Get stock K-line data
  fastify.get<{
    Querystring: { code: string; start?: string; end?: string };
  }>('/stock/kline', async (request, reply) => {
    if (!dbReady) {
      return reply.status(503).send({
        error: 'Database not ready. Run: node scripts/import_stocks_to_sqlite.mjs',
      });
    }
    const { code, start, end } = request.query;
    if (!code) {
      return reply.status(400).send({ error: 'code is required' });
    }
    const data = getStockData(code, start, end);
    if (data.length === 0) {
      return reply.status(404).send({ error: `No data found for code: ${code}` });
    }
    return { code: code.toUpperCase(), count: data.length, data };
  });

  // Search stocks by code (autocomplete)
  fastify.get<{
    Querystring: { q: string };
  }>('/stock/search', async (request) => {
    if (!dbReady) return { results: [] };
    const { q } = request.query;
    if (!q || q.length < 2) return { results: [] };
    return { results: searchCodes(q) };
  });

  // Get backtest results
  fastify.get<{
    Querystring: { strategy?: string };
  }>('/backtest', async (request, reply) => {
    const { strategy = 'momentum' } = request.query;
    try {
      const [nav, returns] = await Promise.all([
        readBacktestCsv(`${strategy}_nav.csv`),
        readBacktestCsv(`${strategy}_monthly_returns.csv`),
      ]);

      // Read optional report markdown
      let report: string | null = null;
      const reportPath = path.join(BACKTEST_DIR, `${strategy}_report.md`);
      if (fs.existsSync(reportPath)) {
        report = fs.readFileSync(reportPath, 'utf8');
      }

      return { strategy, nav, returns, report };
    } catch (err) {
      fastify.log.error(err);
      return reply.status(500).send({ error: 'Failed to read backtest data' });
    }
  });

  // List available strategies, sorted by _nav.csv last modified time (newest first)
  fastify.get('/strategies', async () => {
    try {
      const files = fs.readdirSync(BACKTEST_DIR);
      const strategies = [...new Set(
        files
          .filter((f) => f.endsWith('_nav.csv'))
          .map((f) => f.replace('_nav.csv', ''))
      )];

      // Sort by the mtime of each strategy's _nav.csv, descending
      strategies.sort((a, b) => {
        const mtimeA = fs.statSync(path.join(BACKTEST_DIR, `${a}_nav.csv`)).mtimeMs;
        const mtimeB = fs.statSync(path.join(BACKTEST_DIR, `${b}_nav.csv`)).mtimeMs;
        return mtimeB - mtimeA;
      });

      return { strategies };
    } catch {
      return { strategies: [] };
    }
  });
}
