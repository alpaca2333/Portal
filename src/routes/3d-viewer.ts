import { FastifyInstance, FastifyPluginOptions } from 'fastify';
import multipart from '@fastify/multipart';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const UPLOADS_DIR = path.join(__dirname, '../../public/3d-viewer/models');

if (!fs.existsSync(UPLOADS_DIR)) {
    fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

export default async function viewer3dRoutes(fastify: FastifyInstance, options: FastifyPluginOptions) {
    // 注册 multipart 支持
    await fastify.register(multipart);

    // 获取所有 FBX 文件列表
    fastify.get('/files', async (request, reply) => {
        try {
            console.log('Fetching files from:', UPLOADS_DIR);
            if (!fs.existsSync(UPLOADS_DIR)) {
                console.error('Uploads directory does not exist:', UPLOADS_DIR);
                return [];
            }
            const files = fs.readdirSync(UPLOADS_DIR)
                .filter(file => file.toLowerCase().endsWith('.fbx'))
                .map(file => ({
                    name: file,
                    url: `/3d-viewer/models/${file}`
                }));
            console.log('Found files:', files);
            return files;
        } catch (err) {
            fastify.log.error(err);
            return reply.status(500).send({ error: 'Failed to list files' });
        }
    });

    // 上传 FBX 文件
    fastify.post('/upload', async (request, reply) => {
        console.log('Received upload request');
        try {
            const data = await request.file();
            if (!data) {
                console.error('No file in request');
                return reply.status(400).send({ error: 'No file uploaded' });
            }

            let fileName = data.filename;
            console.log('Uploading file:', fileName);
            const ext = path.extname(fileName);
            const nameWithoutExt = path.basename(fileName, ext);
            
            let targetPath = path.join(UPLOADS_DIR, fileName);
            let counter = 1;

            // 处理同名文件
            while (fs.existsSync(targetPath)) {
                fileName = `${nameWithoutExt}_${counter}${ext}`;
                targetPath = path.join(UPLOADS_DIR, fileName);
                counter++;
            }

            const writeStream = fs.createWriteStream(targetPath);
            await new Promise((resolve, reject) => {
                data.file.pipe(writeStream);
                data.file.on('end', resolve);
                data.file.on('error', reject);
            });

            console.log('File saved to:', targetPath);
            return { 
                success: true, 
                fileName, 
                url: `/3d-viewer/models/${fileName}` 
            };
        } catch (err) {
            console.error('Upload process error:', err);
            return reply.status(500).send({ error: 'Internal upload error' });
        }
    });

    // 删除 FBX 文件
    fastify.delete('/files/:fileName', async (request, reply) => {
        const { fileName } = request.params as { fileName: string };
        const safePath = path.join(UPLOADS_DIR, path.basename(fileName));

        if (!fs.existsSync(safePath)) {
            return reply.status(404).send({ error: 'File not found' });
        }

        try {
            fs.unlinkSync(safePath);
            return { success: true };
        } catch (err) {
            fastify.log.error(err);
            return reply.status(500).send({ error: 'Failed to delete file' });
        }
    });
}
