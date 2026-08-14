import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxying keeps the browser same-origin, so SSE works without any CORS
      // preflight dance and the frontend needs no API base URL configuration.
      '/api': {
        target: 'http://localhost:5001',
        changeOrigin: true,
        // Buffering would defeat the whole point of streaming pipeline stages.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache, no-transform'
            }
          })
        },
      },
    },
  },
})
