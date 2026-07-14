import "./styles.css";
const site = "https://wimpie007120606.github.io/VelaCL/";
const description = "Measured continual-learning experiments";
export const metadata = {
  title: "VelaCL Research Console",
  description,
  openGraph: { title: "VelaCL Research Console", description, url: site, images: [{ url: `${site}og-image.png`, width: 1200, height: 630 }] },
  twitter: { card: "summary_large_image", title: "VelaCL Research Console", description, images: [`${site}og-image.png`] },
};
export default function Layout({children}:{children:React.ReactNode}) { return <html lang="en"><body><noscript>VelaCL research console: measured continual-learning results for a multilingual byte Transformer — random replay reached 30.66% average task macro-F1 versus 3.30% for naive sequential fine-tuning over a six-stage stream.</noscript>{children}</body></html>; }
