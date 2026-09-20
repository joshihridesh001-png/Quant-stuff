/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          950: '#040711',
          900: '#070c18',
          850: '#0b1326',
          800: '#111b36',
          700: '#1c2b54',
        },
        cyan: {
          400: '#22d3ee',
          500: '#06b6d4',
          950: '#083344',
        },
        emerald: {
          400: '#34d399',
          500: '#10b981',
          950: '#064e3b',
        },
        rose: {
          400: '#fb7185',
          500: '#f43f5e',
          950: '#4c0519',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'Courier New', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
