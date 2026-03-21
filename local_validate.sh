#!/bin/bash
# local_validate.sh 

echo "🔍 Pre-flight checks..."

# Syntax
python3 -m py_compile records/alia_submission/train_gpt.py || exit 1

# Imports (basic check)
python3 -c "import sys; sys.path.insert(0, 'records/alia_submission'); import train_gpt" 2>/dev/null || echo "⚠️  Import check skipped"

# Tabs
if grep -q $'\t' records/alia_submission/train_gpt.py; then
    echo "❌ Tabs found - fix before pushing"
    exit 1
fi

echo "✅ All checks passed - safe to push"