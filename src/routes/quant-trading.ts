import { FastifyInstance } from 'fastify';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 读取 Tushare Token
const tokenPath = path.join(__dirname, '../../public/quant-trading/tushare.token');
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

// 缓存股票列表
let stockListCache: any[] = [];
let lastFetchTime = 0;

export default async function quantTradingRoutes(fastify: FastifyInstance) {
    // 搜索股票接口
    fastify.get('/search', async (request, reply) => {
        const { query } = request.query as { query: string };
        console.log(`[Quant API] Search query: "${query}"`);
        if (!query) {
            return { items: [] };
        }

        try {
            // 每天更新一次缓存
            const now = Date.now();
            if (stockListCache.length === 0 || now - lastFetchTime > 24 * 60 * 60 * 1000) {
                console.log('[Quant API] Fetching stock list from Tushare...');
                const data = await fetchTushare('stock_basic', { list_status: 'L' }, 'ts_code,symbol,name');
                // Tushare 返回格式: { fields: [...], items: [[...], [...]] }
                if (data && data.items) {
                    stockListCache = data.items.map((item: any[]) => ({
                        ts_code: item[0],
                        symbol: item[1],
                        name: item[2]
                    }));
                    lastFetchTime = now;
                    console.log(`[Quant API] Cached ${stockListCache.length} stocks`);
                }
            }

            const lowerQuery = query.toLowerCase();
            const results = stockListCache.filter(stock => 
                stock.symbol.includes(lowerQuery) || 
                stock.name.toLowerCase().includes(lowerQuery) ||
                stock.ts_code.toLowerCase().includes(lowerQuery)
            ).slice(0, 10); // 返回前10个匹配项

            console.log(`[Quant API] Found ${results.length} results for "${query}"`);
            return { items: results };
        } catch (error: any) {
            console.error('[Quant API] Search error:', error);
            fastify.log.error(error);
            reply.status(500).send({ error: error.message });
        }
    });

    // 获取 K 线数据接口
    fastify.get('/kline', async (request, reply) => {
        const { ts_code } = request.query as { ts_code: string };
        if (!ts_code) {
            reply.status(400).send({ error: 'ts_code is required' });
            return;
        }

        console.log(`[Quant API] Fetching kline for ${ts_code}`);

        try {
            // 获取过去五年的数据
            const endDate = new Date();
            const startDate = new Date();
            startDate.setFullYear(endDate.getFullYear() - 5);

            const formatDate = (date: Date) => {
                const y = date.getFullYear();
                const m = String(date.getMonth() + 1).padStart(2, '0');
                const d = String(date.getDate()).padStart(2, '0');
                return `${y}${m}${d}`;
            };

            // 判断是否为指数 (简单判断：以 .SH 结尾的 000 开头，或者 .SZ 结尾的 399 开头等，这里为了兼容，如果查不到 daily 就查 index_daily)
            // 但为了简单起见，我们先统一查 daily，如果为空，再尝试查 index_daily
            let data = await fetchTushare(
                'daily', 
                { 
                    ts_code: ts_code,
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
                        ts_code: ts_code,
                        start_date: formatDate(startDate),
                        end_date: formatDate(endDate)
                    }, 
                    'trade_date,open,close,low,high,vol'
                );
            }

            // Tushare 返回的数据是按日期倒序的，我们需要正序给图表
            if (data && data.items) {
                data.items.reverse();
            }

            return data;
        } catch (error: any) {
            console.error(`[Quant API] Error fetching kline for ${ts_code}:`, error);
            fastify.log.error(error);
            reply.status(500).send({ error: error.message });
        }
    });

    // 获取股票基本信息接口
    fastify.get('/basic_info', async (request, reply) => {
        const { ts_code } = request.query as { ts_code: string };
        if (!ts_code) {
            reply.status(400).send({ error: 'ts_code is required' });
            return;
        }

        console.log(`[Quant API] Fetching basic info for ${ts_code}`);

        try {
            // 获取最新一天的基本信息
            const endDate = new Date();
            const startDate = new Date();
            startDate.setDate(endDate.getDate() - 30); // 往前推30天，确保能取到最近一个交易日的数据

            const formatDate = (date: Date) => {
                const y = date.getFullYear();
                const m = String(date.getMonth() + 1).padStart(2, '0');
                const d = String(date.getDate()).padStart(2, '0');
                return `${y}${m}${d}`;
            };

            // 尝试获取股票每日指标
            let data = await fetchTushare(
                'daily_basic', 
                { 
                    ts_code: ts_code,
                    start_date: formatDate(startDate),
                    end_date: formatDate(endDate)
                }, 
                'trade_date,turnover_rate,pe,pb,total_share,float_share,total_mv,circ_mv'
            );

            // 如果没有数据，可能是指数，尝试获取指数每日指标
            if (!data || !data.items || data.items.length === 0) {
                console.log(`[Quant API] No daily_basic data found for ${ts_code}, trying index_dailybasic...`);
                data = await fetchTushare(
                    'index_dailybasic', 
                    { 
                        ts_code: ts_code,
                        start_date: formatDate(startDate),
                        end_date: formatDate(endDate)
                    }, 
                    'trade_date,total_mv,float_mv,total_share,float_share,turnover_rate,pe,pb'
                );
            }

            // 返回最新的一条数据
            if (data && data.items && data.items.length > 0) {
                // Tushare 返回的数据通常是按日期倒序的，第一条就是最新的
                return { fields: data.fields, item: data.items[0] };
            }

            return { fields: [], item: null };
        } catch (error: any) {
            console.error(`[Quant API] Error fetching basic info for ${ts_code}:`, error);
            fastify.log.error(error);
            reply.status(500).send({ error: error.message });
        }
    });
}