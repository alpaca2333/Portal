import Fastify from 'fastify';
import fastifyStatic from '@fastify/static';
import path from 'path';
import { fileURLToPath } from 'url';
import stockDataRoutes from './routes/stock-data.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const fastify = Fastify({
  logger: true
});

// 注册股票数据路由
fastify.register(stockDataRoutes, { prefix: '/api/quant' });

// 注册静态文件服务，指向 public 目录
fastify.register(fastifyStatic, {
  root: path.join(__dirname, '../public'),
  prefix: '/', // 默认访问根路径
});

// 启动服务器，监听 0.0.0.0:80
const start = async () => {
  try {
    await fastify.listen({ 
      port: 8080, 
      host: '0.0.0.0' 
    });
    console.log(`Server is running at http://0.0.0.0:8080`);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

start();