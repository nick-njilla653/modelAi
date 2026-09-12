import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * En développement, le serveur Vite sert l'interface sur :5173 et relaie les
 * appels /api vers FastAPI sur :8000 — pas de CORS à configurer.
 *
 * En production, `npm run build` produit `dist/`, servi directement par FastAPI
 * sous /ui. La base est donc relative : l'interface fonctionne aussi bien
 * derrière le proxy de développement que montée sur un sous-chemin.
 */
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Les réponses juridiques sont longues : on garde les sources pour pouvoir
    // déboguer le rendu Markdown sur un cas réel.
    sourcemap: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
