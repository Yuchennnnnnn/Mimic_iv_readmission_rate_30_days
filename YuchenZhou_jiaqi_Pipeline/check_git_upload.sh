#!/bin/bash
# Git上传前检查脚本 - 确保不会上传大文件

echo "=========================================================================="
echo "Git上传前检查"
echo "=========================================================================="
echo ""

# 检查是否在正确的目录
if [ ! -f ".gitignore" ]; then
    echo "❌ 错误: 请在YuchenZhou_jiaqi_Pipeline目录下运行此脚本"
    exit 1
fi

echo "✓ 在正确的目录"
echo ""

# 检查.gitignore文件
echo "=========================================================================="
echo "检查.gitignore配置"
echo "=========================================================================="
if grep -q "output/" .gitignore && grep -q "*.pkl" .gitignore; then
    echo "✓ .gitignore正确配置"
else
    echo "⚠️ 警告: .gitignore可能配置不正确"
    echo "请确保包含以下行:"
    echo "  output/"
    echo "  *.pkl"
    echo "  *.parquet"
fi
echo ""

# 检查将要上传的文件
echo "=========================================================================="
echo "检查Git状态"
echo "=========================================================================="

# 如果还没有git init
if [ ! -d ".git" ]; then
    echo "⚠️ 还未初始化Git仓库"
    echo "运行: git init"
    echo ""
else
    # 检查staged files
    echo "已暂存的文件:"
    git diff --cached --name-only
    echo ""
    
    # 检查大文件
    echo "检查大文件 (>50MB)..."
    LARGE_FILES=$(git ls-files | xargs ls -lh 2>/dev/null | awk '{if($5 ~ /[0-9]+M/ && $5+0 > 50) print $5, $9}')
    
    if [ -z "$LARGE_FILES" ]; then
        echo "✓ 没有发现>50MB的大文件"
    else
        echo "⚠️ 发现大文件:"
        echo "$LARGE_FILES"
        echo ""
        echo "这些文件不应该上传到GitHub！"
        echo "请运行: git rm --cached <filename>"
    fi
fi
echo ""

# 检查output目录
echo "=========================================================================="
echo "检查output目录"
echo "=========================================================================="
if [ -d "output" ]; then
    OUTPUT_SIZE=$(du -sh output 2>/dev/null | cut -f1)
    echo "output目录大小: $OUTPUT_SIZE"
    echo ""
    
    # 检查output是否在git中
    if git ls-files | grep -q "^output/"; then
        echo "❌ 错误: output/目录中的文件被Git追踪！"
        echo "请运行: git rm -r --cached output/"
    else
        echo "✓ output/目录未被Git追踪"
    fi
else
    echo "⚠️ output目录不存在"
fi
echo ""

# 检查预期的文件结构
echo "=========================================================================="
echo "检查项目文件结构"
echo "=========================================================================="

EXPECTED_DIRS=(
    "preprocessing/scripts"
    "preprocessing"
    "training"
    "testing"
)

for dir in "${EXPECTED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir"
    else
        echo "⚠️ $dir (不存在)"
    fi
done
echo ""

# 估算上传大小
echo "=========================================================================="
echo "估算上传大小"
echo "=========================================================================="

if [ -d ".git" ]; then
    echo "Git仓库统计:"
    git count-objects -vH 2>/dev/null
    echo ""
    
    REPO_SIZE=$(git count-objects -vH 2>/dev/null | grep "size-pack" | awk '{print $2}')
    echo "预计上传大小: $REPO_SIZE"
    echo ""
    
    if [[ "$REPO_SIZE" =~ "GiB" ]]; then
        echo "❌ 警告: 仓库大小>1GB, GitHub会拒绝推送！"
    elif [[ "$REPO_SIZE" =~ "MiB" ]]; then
        SIZE_NUM=$(echo "$REPO_SIZE" | grep -o "[0-9.]*")
        if (( $(echo "$SIZE_NUM > 500" | bc -l) )); then
            echo "⚠️ 警告: 仓库大小>500MB, 建议精简"
        else
            echo "✓ 仓库大小合理"
        fi
    else
        echo "✓ 仓库大小很小"
    fi
fi
echo ""

# 推荐命令
echo "=========================================================================="
echo "推荐的Git命令"
echo "=========================================================================="
echo "如果一切正常，运行以下命令上传:"
echo ""
echo "  git add ."
echo "  git commit -m \"Add MIMIC-IV preprocessing pipeline\""
echo "  git remote add origin https://github.com/Yuchennnnnnn/Mimic_iv_readmission_rate_30_days.git"
echo "  git push -u origin main"
echo ""
echo "如果发现大文件，先移除:"
echo "  git rm --cached -r output/"
echo "  git commit --amend"
echo ""
echo "=========================================================================="
