import Fastify from 'fastify';
import fastifyStatic from '@fastify/static';
import path from 'path';
import { fileURLToPath } from 'url';
import quantTradingRoutes from './routes/quant-trading.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const fastify = Fastify({
  logger: true
});

// 注册量化交易路由
fastify.register(quantTradingRoutes, { prefix: '/api/quant' });

// 注册静态文件服务，指向 public 目录
fastify.register(fastifyStatic, {
  root: path.join(__dirname, '../public'),
  prefix: '/', // 默认访问根路径
});

// 启动服务器，监听 0.0.0.0:8080
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