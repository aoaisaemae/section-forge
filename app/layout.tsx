import type { Metadata } from "next";
import "./globals.css";
import "./workflow.css";
import "./interactions.css";
import "./typology.css";
import "./descriptors.css";
import "./primary-lines.css";
import "./palette.css";
import "./photo-model.css";

export const metadata: Metadata = {
  title: "Section Forge",
  description: "A depth-map studio for constructing CT volumes and smoothed SubD massing.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
