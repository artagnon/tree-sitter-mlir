'use strict';

import * as cp from "node:child_process";
import * as process from "node:process";
import * as path from "node:path";
import * as os from "node:os";
import { glob } from "glob";

const dialects = {
  "Builtin": 100,
  "Func": 100,
  "Arith": 72,
  "Math": 87,
  "ControlFlow": 100,
  "SCF": 73,
  "Memref": 69,
  "Tensor": 68,
  "Affine": 82,
  "Linalg": 37,
};

let mlir_testdir = path.join(os.homedir(), "src", "llvm", "mlir", "test", "Dialect");

function bench_dialect(dialect, min_pct) {
  let testdir = path.join(mlir_testdir, dialect);
  let testfiles = glob.sync(`${testdir}/*.mlir`, { ignore: [`${testdir}/invalid.mlir`] });
  let child = cp.spawn("npx", ["tree-sitter", "parse", "-q", "-s", ...testfiles],
    { cwd: process.cwd() });
  let output = "";
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (data) => output += data);
  child.on("close", () => {
    let match = output.match(/success percentage: (\d+\.\d+)%/i);
    let pass_pct = parseFloat(match[1]);
    if (pass_pct < min_pct) {
      console.log("%s, %f%% passed; minimum required is %d%%", dialect, pass_pct, min_pct);
      process.exit(1);
    }
    console.log("%s, %f%% passed", dialect, pass_pct);
  });
  child.on("error", (err) => console.log(err));
}

for (const [k, v] of Object.entries(dialects)) {
  bench_dialect(k, v);
}
