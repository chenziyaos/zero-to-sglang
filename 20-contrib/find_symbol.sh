#!/usr/bin/env bash
# 给一个符号，回答读源码时最先要问的三件事：
#   1. 定义在哪（def / class / 赋值）
#   2. 谁在调它（排除定义行本身和测试）
#   3. 它所在文件的镜像单测路径，以及仓库里有没有、谁在测它
#
# 用法：bash find_symbol.sh <符号> [sglang 仓库根，默认 ../../sglang]
#   bash find_symbol.sh _calculate_utilization ~/sglang
#   bash find_symbol.sh match_prefix
# 只用 grep，不依赖 ripgrep / ctags。
set -euo pipefail

sym=${1:?用法: find_symbol.sh <符号> [仓库根]}
repo=${2:-$(dirname "$0")/../../sglang}
cd "$repo"
src=python/sglang
[ -d "$src" ] || { echo "在 $repo 下找不到 $src，第二个参数给 sglang 仓库根" >&2; exit 2; }

hr() { printf '\n== %s\n' "$1"; }

hr "定义"
defs=$(grep -rnE --include='*.py' "^[[:space:]]*(async[[:space:]]+)?(def|class)[[:space:]]+${sym}\b|^[[:space:]]*${sym}[[:space:]]*(:[^=]*)?=[^=]" "$src" || true)
if [ -z "$defs" ]; then
  echo "  没找到 def/class/赋值；可能是动态属性，用 grep -rn '${sym}' $src 自己看"
else
  echo "$defs" | sed 's/^/  /'
fi
defs=$(printf '%s\n' "$defs" | grep -v '^python/sglang/test/' || true)

hr "调用点（真正的调用 ${sym}(…；不含定义、注释和测试）"
calls=$(grep -rnE --include='*.py' "\b${sym}[[:space:]]*\(" "$src" \
  | grep -vE "(def|class)[[:space:]]+${sym}\b" \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*#' \
  | grep -vE '^python/sglang/test/' || true)
n=$(printf '%s' "$calls" | grep -c . || true)
echo "  共 $n 处"
printf '%s\n' "$calls" | head -30 | sed 's/^/  /'
if [ "$n" -gt 30 ]; then echo "  ……其余 $((n - 30)) 处略，加 | less 自己翻"; fi

hr "镜像单测（test/registered/unit/README.md 的规则）"
echo "$defs" | cut -d: -f1 | sort -u | while read -r f; do
  [ -n "$f" ] || continue
  case "$f" in
    python/sglang/srt/*.py)
      rel=${f#python/sglang/srt/}
      t="test/registered/unit/$(dirname "$rel")/test_$(basename "$rel")"
      t=${t/\/.\//\/}
      if [ -f "$t" ]; then
        echo "  $f -> $t  [已有]"
      else
        near=$(ls "$(dirname "$t")"/test_"$(basename "$rel" .py)"*.py 2>/dev/null | head -3 | tr '\n' ' ' || true)
        if [ -n "$near" ]; then echo "  $f -> $t  [没有同名，近似：$near]"
        else echo "  $f -> $t  [没有，新建]"; fi
      fi
      ;;
    *) echo "  $f 不在 srt 下，没有镜像规则" ;;
  esac
done

hr "已有测试里提到它的地方"
grep -rlE --include='*.py' "\b${sym}\b" test 2>/dev/null | head -10 | sed 's/^/  /' || true
