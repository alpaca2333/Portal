import { FastifyInstance } from 'fastify';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { readCache, writeCache } from './disk-cache.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 读取 Tushare Token
const tokenPath = path.join(__dirname, '../../public/stock-data/tushare.token');
let tushareToken = '';
try {
    tushareToken = fs.readFileSync(tokenPath, 'utf-8').trim();
} catch (error) {
    console.error('Failed to read Tushare token:', error);
}

const TUSHARE_API_URL = 'https://api.tushare.pro';

async function fetchTushare(apiName: string, params: any = {}, fields: string = '') {
    if (!tushareToken) {
        console.error('[Tushare] Token not found');
        throw new Error('Tushare token not found');
    }

    console.log(`[Tushare] Requesting API: ${apiName}`, { params, fields });

    const response = await fetch(TUSHARE_API_URL, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            api_name: apiName,
            token: tushareToken,
            params: params,
            fields: fields
        })
    });

    if (!response.ok) {
        console.error(`[Tushare] HTTP Error: ${response.status} ${response.statusText}`);
        throw new Error(`Tushare API error: ${response.statusText}`);
    }

    const data = await response.json();
    console.log(`[Tushare] Response for ${apiName}:`, JSON.stringify(data).substring(0, 200) + '...');

    if (data.code !== 0) {
        console.error(`[Tushare] API Error: ${data.msg}`);
        throw new Error(`Tushare API error: ${data.msg}`);
    }

    return data.data;
}

// ─── helpers ────────────────────────────────────────────────────────────────

const formatDate = (date: Date): string => {
    const y = date.getFullYear();
    const m = String(date.getMonth() + 1).padStart(2, '0');
    const d = String(date.getDate()).padStart(2, '0');
    return `${y}${m}${d}`;
};

/** Fire-and-forget background refresh. Errors are only logged. */
function refreshInBackground(key: string, fetcher: () => Promise<any>): void {
    fetcher()
        .then(data => { writeCache(key, data); })
        .catch(err => { console.warn(`[Cache] Background refresh failed for "${key}":`, err); });
}

// ─── stock-list fetcher ──────────────────────────────────────────────────────

async function fetchAndFormatStockList(): Promise<any[]> {
    const data = await fetchTushare('stock_basic', { list_status: 'L' }, 'ts_code,symbol,name');
    if (data && data.items) {
        return data.items.map((item: any[]) => ({
            ts_code: item[0],
            symbol: item[1],
            name: item[2]
        }));
    }
    return [];
}

// ─── kline fetcher ───────────────────────────────────────────────────────────

async function fetchAndFormatKline(ts_code: string): Promise<any> {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setFullYear(endDate.getFullYear() - 5);

    let data = await fetchTushare(
        'daily',
        {
            ts_code,
            start_date: formatDate(startDate),
            end_date: formatDate(endDate)
        },
        'trade_date,open,close,low,high,vol'
    );

    if (!data || !data.items || data.items.length === 0) {
        console.log(`[Quant API] No daily data found for ${ts_code}, trying index_daily...`);
        data = await fetchTushare(
            'index_daily',
            {
                ts_code,
                start_date: formatDate(startDate),
                end_date: formatDate(endDate)
            },
            'trade_date,open,close,low,high,vol'
        );
    }

    // Tushare 返回的是倒序，正序给图表
    if (data && data.items) {
        data.items.reverse();
    }

    return data;
}

// ─── basic_info fetcher ──────────────────────────────────────────────────────

async function fetchAndFormatBasicInfo(ts_code: string): Promise<any> {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(endDate.getDate() - 30);

    let data = await fetchTushare(
        'daily_basic',
        {
            ts_code,
            start_date: formatDate(startDate),
            end_date: formatDate(endDate)
        },
        'trade_date,turnover_rate,pe,pb,total_share,float_share,total_mv,circ_mv'
    );

    if (!data || !data.items || data.items.length === 0) {
        console.log(`[Quant API] No daily_basic data found for ${ts_code}, trying index_dailybasic...`);
        data = await fetchTushare(
            'index_dailybasic',
            {
                ts_code,
                start_date: formatDate(startDate),
                end_date: formatDate(endDate)
            },
            'trade_date,total_mv,float_mv,total_share,float_share,turnover_rate,pe,pb'
        );
    }

    if (data && data.items && data.items.length > 0) {
        return { fields: data.fields, item: data.items[0] };
    }
    return { fields: [], item: null };
}

// ─── routes ──────────────────────────────────────────────────────────────────

export default async function stockDataRoutes(fastify: FastifyInstance) {

    // ── /search ──────────────────────────────────────────────────────────────
    fastify.get('/search', async (request, reply) => {
        const { query } = request.query as { query: string };
        console.log(`[Quant API] Search query: "${query}"`);
        if (!query) {
            return { items: [] };
        }

        const CACHE_KEY = 'stock_list';

        try {
            const cached = readCache<any[]>(CACHE_KEY);

            let stockList: any[];

            if (cached === null) {
                // 首次请求，无本地缓存：同步拉取并写入
                console.log('[Quant API] No stock list cache found, fetching from Tushare...');
                stockList = await fetchAndFormatStockList();
                writeCache(CACHE_KEY, stockList);
                console.log(`[Quant API] Fetched and cached ${stockList.length} stocks`);
            } else {
                // 有缓存（新鲜或过期）：立即使用，过期则后台刷新
                stockList = cached.data;
                if (cached.isStale) {
                    console.log('[Quant API] Stock list cache is stale, triggering background refresh...');
                    refreshInBackground(CACHE_KEY, fetchAndFormatStockList);
                } else {
                    console.log(`[Quant API] Using fresh stock list cache (${stockList.length} stocks)`);
                }
            }

            const lowerQuery = query.toLowerCase();
            const results = stockList.filter(stock =>
                stock.symbol.includes(lowerQuery) ||
                stock.name.toLowerCase().includes(lowerQuery) ||
                stock.ts_code.toLowerCase().includes(lowerQuery)
            ).slice(0, 10);

            console.log(`[Quant API] Found ${results.length} results for "${query}"`);
            return { items: results };

        } catch (error: any) {
            console.error('[Quant API] Search error:', error);
            fastify.log.error(error);
            reply.status(500).send({ error: error.message });
        }
    });

    // ── /kline ───────────────────────────────────────────────────────────────
    fastify.get('/kline', async (request, reply) => {
        const { ts_code } = request.query as { ts_code: string };
        if (!ts_code) {
            reply.status(400).send({ error: 'ts_code is required' });
            return;
        }

        console.log(`[Quant API] Fetching kline for ${ts_code}`);
        const CACHE_KEY = `kline_${ts_code}`;

        try {
            const cached = readCache<any>(CACHE_KEY);

            if (cached === null) {
                // 首次请求，无缓存：同步拉取
                console.log(`[Quant API] No kline cache for ${ts_code}, fetching from Tushare...`);
                const data = await fetchAndFormatKline(ts_code);
                writeCache(CACHE_KEY, data);
                return data;
            }

            // 有缓存：立即返回
            if (cached.isStale) {
                console.log(`[Quant API] Kline cache for ${ts_code} is stale, serving cache and refreshing in background...`);
                refreshInBackground(CACHE_KEY, () => fetchAndFormatKline(ts_code));
            } else {
                console.log(`[Quant API] Serving fresh kline cache for ${ts_code}`);
            }
            return cached.data;

        } catch (error: any) {
            console.error(`[Quant API] Error fetching kline for ${ts_code}:`, error);
            fastify.log.error(error);
            reply.status(500).send({ error: error.message });
        }
    });

    // ── /basic_info ──────────────────────────────────────────────────────────
    fastify.get('/basic_info', async (request, reply) => {
        const { ts_code } = request.query as { ts_code: string };
        if (!ts_code) {
            reply.status(400).send({ error: 'ts_code is required' });
            return;
        }

        console.log(`[Quant API] Fetching basic info for ${ts_code}`);
        const CACHE_KEY = `basic_info_${ts_code}`;

        try {
            const cached = readCache<any>(CACHE_KEY);

            if (cached === null) {
                // 首次请求，无缓存：同步拉取
                console.log(`[Quant API] No basic_info cache for ${ts_code}, fetching from Tushare...`);
                const data = await fetchAndFormatBasicInfo(ts_code);
                writeCache(CACHE_KEY, data);
                return data;
            }

            // 有缓存：立即返回
            if (cached.isStale) {
                console.log(`[Quant API] Basic info cache for ${ts_code} is stale, serving cache and refreshing in background...`);
                refreshInBackground(CACHE_KEY, () => fetchAndFormatBasicInfo(ts_code));
            } else {
                console.log(`[Quant API] Serving fresh basic info cache for ${ts_code}`);
            }
            return cached.data;

        } catch (error: any) {
            console.error(`[Quant API] Error fetching basic info for ${ts_code}:`, error);
            fastify.log.error(error);
            reply.status(500).send({ error: error.message });
        }
    });
}
