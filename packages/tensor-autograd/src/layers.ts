/**
 * The public `nn` namespace: the core layer/loss set (`./nn.ts`) plus the
 * transformer blocks (`./transformer.ts`, issue #123). A separate barrel so
 * `transformer.ts` can import from `nn.ts` without an import cycle.
 */
export * from "./nn.ts";
export * from "./transformer.ts";
