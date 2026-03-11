import fs from 'fs';
const token = fs.readFileSync('/projects/portal/public/quant-trading/tushare.token', 'utf-8').trim();
const TUSHARE_API_URL = 'https://api.tushare.pro';

async function test() {
    const response = await fetch(TUSHARE_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            api_name: 'stock_basic',
            token: token,
            params: { list_status: 'L' },
            fields: 'ts_code,symbol,name'
        })
    });
    const data = await response.json();
    console.log(JSON.stringify(data).substring(0, 500));
}
test();
