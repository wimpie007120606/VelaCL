import "./styles.css";
export const metadata = { title: "VelaCL Research Console", description: "Measured continual-learning experiments" };
export default function Layout({children}:{children:React.ReactNode}) { return <html lang="en"><body><noscript>VelaCL research console: measured continual-learning results for a multilingual byte Transformer — random replay reached 30.66% average task macro-F1 versus 3.30% for naive sequential fine-tuning over a six-stage stream.</noscript>{children}</body></html>; }
