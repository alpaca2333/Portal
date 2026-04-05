import Fastify from 'fastify';
import fastifyStatic from '@fastify/static';
import path from 'path';
import { fileURLToPath } from 'url';
import stockDataRoutes from './routes/stock-data.js';
import quantRoutes from './routes/quant.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const port = Number(process.env.PORT || 8080);
const host = process.env.HOST || '0.0.0.0';

const fastify = Fastify({ logger: true });

fastify.register(stockDataRoutes, { prefix: '/api/stock-data' });
fastify.register(quantRoutes, { prefix: '/api/quant' });
fastify.register(fastifyStatic, {
  root: path.join(__dirname, '../public'),
  prefix: '/',
});

let shuttingDown = false;

const shutdown = async (signal: string, err?: unknown) => {
  if (shuttingDown) return;
  shuttingDown = true;

  if (err) {
    fastify.log.error({ err }, `Fatal signal received: ${signal}`);
  } else {
    fastify.log.info(`Signal received: ${signal}, shutting down gracefully`);
  }

  try {
    await fastify.close();
  } catch (closeErr) {
    fastify.log.error({ err: closeErr }, 'Failed to close Fastify cleanly');
    process.exit(1);
  }

  process.exit(err ? 1 : 0);
};

process.on('SIGINT', () => { void shutdown('SIGINT'); });
process.on('SIGTERM', () => { void shutdown('SIGTERM'); });
process.on('uncaughtException', (err) => { void shutdown('uncaughtException', err); });
process.on('unhandledRejection', (reason) => { void shutdown('unhandledRejection', reason); });

const start = async () => {
  try {
    await fastify.listen({ port, host });
    fastify.log.info(`Server is running at http://${host}:${port}`);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

void start();
