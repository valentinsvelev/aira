import { readFileSync } from 'node:fs';

/** Read a project file by its path relative to web/. */
export const source = (relativePath) => readFileSync(new URL(`../${relativePath}`, import.meta.url), 'utf8');
