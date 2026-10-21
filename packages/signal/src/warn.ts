/**
 * Where warnings go. SciPy's `warnings.warn` becomes `onWarning(message)`; without a callback the message goes to
 * `console.warn`. Pass `() => {}` to silence them. The messages are SciPy's own text.
 */
export interface WarningOptions {
  readonly onWarning?: (message: string) => void;
}

export function emitWarning(options: WarningOptions | undefined, message: string): void {
  if (options?.onWarning) options.onWarning(message);
  else console.warn(message);
}
