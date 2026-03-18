import { FastifyInstance } from 'fastify';
import { parse } from 'csv-parse';
import * as fs from 'fs';
import * as path from 'path';
import * as readline from 'readline';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Data files are stored inside the portal project under data/quant/
const PROJECT_ROOT = path.resolve(__dirname, '../../');
const DATA_FILE = path.join(PROJECT_ROOT, 'data/quant/processed/all_stocks_daily.csv');
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

// ─── In-memory stock database ─────────────────────────────────────────────────

/** code (uppercase, e.g. "SH600519") → sorted KlinePoint[] */
const stockDB = new Map<string, KlinePoint[]>();

let dbReady = false;
let dbLoadedRows = 0;
let dbLoadStartTime = 0;

/**
 * Load the entire CSV into memory once at startup.
 * ~770 MB / 14.2 M rows → ~1-2 GB RSS, loads in ~20-40 s depending on disk.
 */
async function preloadStockData(log: (msg: string) => void): Promise<void> {
  dbLoadStartTime = Date.now();
  log(`[Quant DB] Starting full CSV preload: ${DATA_FILE}`);

  const BATCH_SIZE = 20_000;

  return new Promise((resolve, reject) => {
    const stream = fs.createReadStream(DATA_FILE, { encoding: 'utf8' });
    const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

    let isHeader = true;
    const colIndex: Record<string, number> = {};
    let iCode = 0, iDate = 0, iOpen = 0, iHigh = 0, iLow = 0, iClose = 0, iVolume = 0;

    let batchCount = 0;

    rl.on('line', (line) => {
      if (!line || line.length === 0) return;

      if (isHeader) {
        isHeader = false;
        line.split(',').forEach((c, i) => { colIndex[c.trim()] = i; });
        iCode   = colIndex['code'];
        iDate   = colIndex['date'];
        iOpen   = colIndex['open'];
        iHigh   = colIndex['high'];
        iLow    = colIndex['low'];
        iClose  = colIndex['close'];
        iVolume = colIndex['volume'];
        return;
      }

      const cols = line.split(',');
      const rawCode = cols[iCode]?.trim();
      if (!rawCode) return;

      const code = rawCode.toUpperCase();
      const date = cols[iDate]?.trim();
      if (!date) return;

      const point: KlinePoint = {
        date,
        open:   parseFloat(cols[iOpen]),
        high:   parseFloat(cols[iHigh]),
        low:    parseFloat(cols[iLow]),
        close:  parseFloat(cols[iClose]),
        volume: parseFloat(cols[iVolume]),
      };

      let arr = stockDB.get(code);
      if (!arr) { arr = []; stockDB.set(code, arr); }
      arr.push(point);
      dbLoadedRows++;
      batchCount++;

      if (batchCount >= BATCH_SIZE) {
        batchCount = 0;
        rl.pause();
        const elapsed = ((Date.now() - dbLoadStartTime) / 1000).toFixed(1);
        log(`[Quant DB] Loaded ${dbLoadedRows.toLocaleString()} rows, ${stockDB.size} stocks (${elapsed}s)`);
        setImmediate(() => rl.resume());
      }
    });

    rl.on('close', () => {
      const elapsed = ((Date.now() - dbLoadStartTime) / 1000).toFixed(1);
      log(
        `[Quant DB] ✅ Preload complete: ${dbLoadedRows.toLocaleString()} rows, ` +
        `${stockDB.size} stocks, ${elapsed}s`
      );
      dbReady = true;
      resolve();
    });

    rl.on('error', (err) => {
      log(`[Quant DB] readline error: ${err.message}`);
      reject(err);
    });
    stream.on('error', (err) => {
      log(`[Quant DB] stream error: ${err.message}`);
      reject(err);
    });
  });
}

// ─── Query helpers (all in-memory) ───────────────────────────────────────────

function getStockData(code: string, start?: string, end?: string): KlinePoint[] {
  const rows = stockDB.get(code.toUpperCase());
  if (!rows) return [];
  if (!start && !end) return rows;
  return rows.filter((r) => {
    if (start && r.date < start) return false;
    if (end   && r.date > end)   return false;
    return true;
  });
}

function getAllCodes(): string[] {
  return [...stockDB.keys()].sort();
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
  // Health check
  fastify.get('/health', async () => ({ status: 'ok' }));

  // Loading status
  fastify.get('/status', async () => ({
    ready: dbReady,
    loadedRows: dbLoadedRows,
    stockCount: stockDB.size,
    elapsedMs: Date.now() - dbLoadStartTime,
  }));

  // Get stock K-line data
  fastify.get<{
    Querystring: { code: string; start?: string; end?: string };
  }>('/stock/kline', async (request, reply) => {
    if (!dbReady) {
      return reply.status(503).send({
        error: 'Data is still loading, please wait…',
        loadedRows: dbLoadedRows,
        stockCount: stockDB.size,
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

    const query = q.toUpperCase();
    const results = getAllCodes()
      .filter((c) => c.includes(query))
      .slice(0, 20);
    return { results };
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
      return { strategy, nav, returns };
    } catch (err) {
      fastify.log.error(err);
      return reply.status(500).send({ error: 'Failed to read backtest data' });
    }
  });

  // List available strategies
  fastify.get('/strategies', async () => {
    try {
      const files = fs.readdirSync(BACKTEST_DIR);
      const strategies = [...new Set(
        files
          .filter((f) => f.endsWith('_nav.csv'))
          .map((f) => f.replace('_nav.csv', ''))
      )];
      return { strategies };
    } catch {
      return { strategies: [] };
    }
  });

  // Start preloading data in background (only once)
  if (!dbReady && dbLoadStartTime === 0) {
    preloadStockData((msg) => fastify.log.info(msg)).catch((err) => {
      fastify.log.error('[Quant DB] Preload failed:', err);
    });
  }
}
