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
const DB_FILE     = path.join(PROJECT_ROOT, 'data/quant/data/quant.db');
const BACKTEST_DIR = path.join(PROJECT_ROOT, 'data/quant/backtest');

// ─── Types ───────────────────────────────────────────────────────────────────

interface KlinePoint {
  date: string;      // YYYY-MM-DD for frontend display
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface SearchResult {
  code: string;   // display code: SH600519
  name: string;   // stock name: 贵州茅台
}

// ─── Code format conversion ───────────────────────────────────────────────────
// Frontend uses "SH600519" / "SZ000001" / "BJ830799"
// Database uses "600519.SH" / "000001.SZ" / "830799.BJ"

function toTsCode(displayCode: string): string {
  const upper = displayCode.toUpperCase().trim();
  // Already in ts_code format (contains a dot)
  if (upper.includes('.')) return upper;
  // SH600519 -> 600519.SH
  const prefix = upper.slice(0, 2);
  const num = upper.slice(2);
  return `${num}.${prefix}`;
}

function toDisplayCode(tsCode: string): string {
  // 600519.SH -> SH600519
  const [num, exchange] = tsCode.split('.');
  return `${exchange}${num}`;
}

function formatDate(yyyymmdd: string): string {
  // 20060104 -> 2006-01-04
  return `${yyyymmdd.slice(0, 4)}-${yyyymmdd.slice(4, 6)}-${yyyymmdd.slice(6, 8)}`;
}

function toDbDate(isoDate: string): string {
  // 2006-01-04 -> 20060104
  return isoDate.replace(/-/g, '');
}

// ─── SQLite connection ────────────────────────────────────────────────────────

let db: DatabaseType | null = null;
let dbReady = false;
let stockCount = 0;

function openDb(log: (msg: string) => void): void {
  if (!fs.existsSync(DB_FILE)) {
    log(`[Quant DB] ⚠️  SQLite DB not found: ${DB_FILE}`);
    log(`[Quant DB]    Run: python scripts/build_db.py`);
    return;
  }

  try {
    db = new Database(DB_FILE, { readonly: true });
    db.pragma('journal_mode = WAL');
    db.pragma('cache_size = -32768');  // 32 MB page cache
    db.pragma('temp_store = MEMORY');

    // Count distinct stocks
    const row = db.prepare('SELECT COUNT(DISTINCT ts_code) AS cnt FROM stock_daily').get() as { cnt: number };
    stockCount = row.cnt;
    dbReady = true;
    log(`[Quant DB] ✅ SQLite ready: ${stockCount.toLocaleString()} stocks — ${DB_FILE}`);
  } catch (err: any) {
    log(`[Quant DB] ❌ Failed to open SQLite: ${err.message}`);
  }
}

// ─── Query helpers ────────────────────────────────────────────────────────────

function getStockData(displayCode: string, start?: string, end?: string): KlinePoint[] {
  if (!db) return [];

  const tsCode = toTsCode(displayCode);
  const startDate = start ? toDbDate(start) : undefined;
  const endDate = end ? toDbDate(end) : undefined;

  let sql = 'SELECT trade_date, open, high, low, close, vol FROM stock_daily WHERE ts_code = ?';
  const params: any[] = [tsCode];

  if (startDate) { sql += ' AND trade_date >= ?'; params.push(startDate); }
  if (endDate) { sql += ' AND trade_date <= ?'; params.push(endDate); }
  sql += ' ORDER BY trade_date';

  const rows = db.prepare(sql).all(...params) as { trade_date: string; open: number; high: number; low: number; close: number; vol: number }[];

  return rows.map(r => ({
    date: formatDate(r.trade_date),
    open: r.open,
    high: r.high,
    low: r.low,
    close: r.close,
    volume: r.vol,
  }));
}

function getStockName(displayCode: string): string | null {
  if (!db) return null;
  const tsCode = toTsCode(displayCode);
  const row = db.prepare('SELECT name FROM stock_info WHERE ts_code = ?').get(tsCode) as { name: string } | undefined;
  return row?.name ?? null;
}

function searchCodes(query: string): SearchResult[] {
  if (!db) return [];
  const upper = query.toUpperCase().trim();

  // Search by ts_code pattern or by stock name
  const rows = db.prepare(`
    SELECT ts_code, name FROM stock_info
    WHERE ts_code LIKE ? OR name LIKE ?
    ORDER BY ts_code LIMIT 20
  `).all(`%${upper}%`, `%${query}%`) as { ts_code: string; name: string }[];

  return rows.map(r => ({
    code: toDisplayCode(r.ts_code),
    name: r.name,
  }));
}

// ─── Detail query (single day, all columns) ──────────────────────────────────

function getStockDetail(displayCode: string, date: string): Record<string, any> | null {
  if (!db) return null;
  const tsCode = toTsCode(displayCode);
  const dbDate = toDbDate(date);

  const row = db.prepare(`
    SELECT d.*,
           i1.industry_name AS sw_l1_name,
           i2.industry_name AS sw_l2_name
    FROM stock_daily d
    LEFT JOIN industry_info i1 ON d.sw_l1 = i1.industry_code
    LEFT JOIN industry_info i2 ON d.sw_l2 = i2.industry_code
    WHERE d.ts_code = ? AND d.trade_date = ?
  `).get(tsCode, dbDate) as Record<string, any> | undefined;

  if (!row) return null;

  // Format for frontend
  return {
    ts_code: row.ts_code,
    trade_date: formatDate(String(row.trade_date)),
    // Price & volume
    open: row.open, high: row.high, low: row.low, close: row.close,
    pre_close: row.pre_close, change: row.change, pct_chg: row.pct_chg,
    vol: row.vol, amount: row.amount,
    // Valuation
    turnover_rate: row.turnover_rate, turnover_rate_f: row.turnover_rate_f,
    volume_ratio: row.volume_ratio,
    pe: row.pe, pe_ttm: row.pe_ttm, pb: row.pb,
    ps: row.ps, ps_ttm: row.ps_ttm,
    dv_ratio: row.dv_ratio, dv_ttm: row.dv_ttm,
    total_mv: row.total_mv, circ_mv: row.circ_mv,
    total_share: row.total_share, float_share: row.float_share, free_share: row.free_share,
    // Adj factor
    adj_factor: row.adj_factor,
    // Financials – profitability
    eps: row.eps, bps: row.bps, cfps: row.cfps, revenue_ps: row.revenue_ps,
    roe: row.roe, roe_dt: row.roe_dt, roe_waa: row.roe_waa, roe_yearly: row.roe_yearly,
    roa: row.roa, roa_yearly: row.roa_yearly,
    grossprofit_margin: row.grossprofit_margin, netprofit_margin: row.netprofit_margin,
    profit_to_gr: row.profit_to_gr,
    // Financials – solvency & operations
    debt_to_assets: row.debt_to_assets,
    current_ratio: row.current_ratio, quick_ratio: row.quick_ratio,
    inv_turn: row.inv_turn, ar_turn: row.ar_turn,
    ca_turn: row.ca_turn, fa_turn: row.fa_turn, assets_turn: row.assets_turn,
    // YoY growth
    op_yoy: row.op_yoy, ebt_yoy: row.ebt_yoy, tr_yoy: row.tr_yoy,
    or_yoy: row.or_yoy, equity_yoy: row.equity_yoy,
    // Industry
    sw_l1: row.sw_l1, sw_l1_name: row.sw_l1_name,
    sw_l2: row.sw_l2, sw_l2_name: row.sw_l2_name,
    // Status
    is_suspended: row.is_suspended,
  };
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
    stockCount,
  }));

  // Get stock K-line data
  fastify.get<{
    Querystring: { code: string; start?: string; end?: string };
  }>('/stock/kline', async (request, reply) => {
    if (!dbReady) {
      return reply.status(503).send({
        error: 'Database not ready. Run: python scripts/build_db.py',
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
    const name = getStockName(code);
    return { code: code.toUpperCase(), name, count: data.length, data };
  });

  // Search stocks by code or name (autocomplete)
  fastify.get<{
    Querystring: { q: string };
  }>('/stock/search', async (request) => {
    if (!dbReady) return { results: [] };
    const { q } = request.query;
    if (!q || q.length < 1) return { results: [] };
    return { results: searchCodes(q) };
  });

  // Get single-day detail (all fields) for clicked date
  fastify.get<{
    Querystring: { code: string; date: string };
  }>('/stock/detail', async (request, reply) => {
    if (!dbReady) {
      return reply.status(503).send({ error: 'Database not ready' });
    }
    const { code, date } = request.query;
    if (!code || !date) {
      return reply.status(400).send({ error: 'code and date are required' });
    }
    const detail = getStockDetail(code, date);
    if (!detail) {
      return reply.status(404).send({ error: `No data for ${code} on ${date}` });
    }
    return detail;
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

  // Get trade orders for a specific strategy and date
  fastify.get<{
    Querystring: { strategy?: string; date?: string };
  }>('/backtest/trades', async (request, reply) => {
    const { strategy, date } = request.query;
    if (!strategy) {
      return reply.status(400).send({ error: 'strategy is required' });
    }
    try {
      const rows = await readBacktestCsv(`${strategy}-trade.csv`);
      if (!date) {
        return { trades: rows };
      }
      // date can be YYYY-MM-DD or YYYYMMDD
      const normalised = date.replace(/-/g, '');
      const filtered = rows.filter((r: Record<string, string>) => {
        const rd = (r.date || r.trade_date || '').replace(/-/g, '');
        return rd === normalised;
      });
      // Enrich trades with stock names from DB
      const enriched = enrichTradesWithNames(filtered);
      return { trades: enriched };
    } catch (err) {
      fastify.log.error(err);
      return reply.status(500).send({ error: 'Failed to read trade data' });
    }
  });

  // ─── Helper: enrich trade rows with stock names ───────────────────────────
  function enrichTradesWithNames(trades: Record<string, string>[]): Record<string, string>[] {
    if (!db || trades.length === 0) return trades;
    // Collect unique ts_codes
    const codes = [...new Set(trades.map(t => t.ts_code).filter(Boolean))];
    if (codes.length === 0) return trades;
    // Batch query stock names
    const nameMap: Record<string, string> = {};
    const placeholders = codes.map(() => '?').join(',');
    const nameRows = db.prepare(
      `SELECT ts_code, name FROM stock_info WHERE ts_code IN (${placeholders})`
    ).all(...codes) as { ts_code: string; name: string }[];
    nameRows.forEach(r => { nameMap[r.ts_code] = r.name; });
    // Attach name to each trade
    return trades.map(t => ({
      ...t,
      name: nameMap[t.ts_code] || '',
    }));
  }

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
