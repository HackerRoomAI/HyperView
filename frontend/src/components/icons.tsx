"use client";

import hyperviewIcon from "../assets/hyperview-icon.png";

/**
 * Shared icons for HyperView UI, including the generated product mark.
 */

export const HyperViewLogo = ({ className = "w-5 h-5" }: { className?: string }) => (
  // Native images preserve the relative asset URL in standalone Static Spaces.
  // eslint-disable-next-line @next/next/no-img-element
  <img src={hyperviewIcon.src} width={20} height={20} className={className} alt="HyperView" draggable={false} />
);

export const CheckIcon = () => (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
  </svg>
);
