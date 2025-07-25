// apps/web/astro.config.mjs
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  integrations: [
    react(),
    tailwind({
      applyBaseStyles: false,
    })
  ],
  server: {
    port: 4321,
    host: true
  },
  output: 'static',
  build: {
    assets: 'assets'
  },
  vite: {
    optimizeDeps: {
      include: ['react', 'react-dom', 'livekit-client']
    },
    define: {
      global: 'globalThis'
    }
  }
});