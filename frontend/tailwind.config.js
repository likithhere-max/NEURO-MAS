/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        neuro: {
          bg: "#0b1020",
          panel: "#121933",
          panel2: "#171f42",
          border: "#26305c",
          accent: "#5b8cff",
          accent2: "#7ee7d0",
          text: "#e7ecff",
          muted: "#93a0cc",
          warn: "#f4b740",
          high: "#ff8a5b",
          critical: "#ff5c7a",
          ok: "#4fd39a",
        },
      },
    },
  },
  plugins: [],
};
