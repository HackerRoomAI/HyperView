import type { Metadata } from "next";
import hyperviewIcon from "../assets/hyperview-icon.png";
import "./globals.css";

export const metadata: Metadata = {
  title: "HyperView",
  description: "Dataset visualization with hyperbolic embeddings",
  icons: {
    icon: { url: hyperviewIcon.src, sizes: "256x256", type: "image/png" },
    apple: { url: hyperviewIcon.src, sizes: "256x256", type: "image/png" },
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body className="antialiased h-full">{children}</body>
    </html>
  );
}
