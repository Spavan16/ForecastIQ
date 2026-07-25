import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ForecastIQ | AI-Powered Revenue Intelligence Platform",
  // NOTE: this is the page's <meta name="description"> tag -- visible in browser tab
  // previews, page source, and any social/link-preview card if the live demo URL is shared.
  description: "From Marketing Spend to Revenue Certainty — 30, 60, and 90-Day Enterprise Outlook",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-900 text-slate-50 antialiased selection:bg-[#1F7A78]/30 selection:text-[#2A1F18]">
        {children}
      </body>
    </html>
  );
}
