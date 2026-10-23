/** Minimal open book, with short reading lines on both facing pages. */
export function ResearchLogo({ size = 25 }: { size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
    <path d="M16 26c-4.4-3.1-8.8-3.6-12.5-2.5V7.3C8 5.9 12.3 6.9 16 9.6c3.7-2.7 8-3.7 12.5-2.3v16.2C24.8 22.4 20.4 22.9 16 26Z" stroke="currentColor" strokeWidth="1.85" strokeLinejoin="round"/>
    <path d="M16 9.6V26M7.3 11.8l5.4 1.1M7.3 15.8l5.4 1.1M19.3 12.9l5.4-1.1M19.3 16.9l5.4-1.1" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
  </svg>;
}
