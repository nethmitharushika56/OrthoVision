import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [
    react({
      include: /\.[jt]sx?$/,
    }),
  ],
  server: {
    proxy: {
      '/analyze': 'http://localhost:5000',
      '/get_heatmap': 'http://localhost:5000',
      '/health': 'http://localhost:5000',
    },
  },
})
