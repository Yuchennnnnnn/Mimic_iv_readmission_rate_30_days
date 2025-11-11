# 🚀 快速命令参考

## 上传到GitHub（排除大文件）

```bash
cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/proj_v2

# 检查.gitignore是否正确配置
cat YuchenZhou_jiaqi_Pipeline/.gitignore

# 初始化git（如果需要）
git init

# 添加文件（会自动排除output/和大文件）
git add YuchenZhou_jiaqi_Pipeline/

# 检查将要上传的文件（确保没有大文件）
git status

# 查看文件大小
git ls-files | xargs ls -lh | awk '{if($5 ~ /[0-9]+M/ && $5+0 > 50) print $5, $9}'

# 提交
git commit -m "Add MIMIC-IV preprocessing pipeline"

# 添加远程仓库
git remote add origin https://github.com/Yuchennnnnnn/Mimic_iv_readmission_rate_30_days.git

# 推送
git branch -M main
git push -u origin main
```

## 如果意外添加了大文件

```bash
# 从暂存区移除
git reset HEAD YuchenZhou_jiaqi_Pipeline/output/

# 或完全移除
git rm --cached -r YuchenZhou_jiaqi_Pipeline/output/

# 重新提交
git commit --amend
git push origin main --force
```

## 运行预处理

```bash
cd YuchenZhou_jiaqi_Pipeline/preprocessing

# 一键运行（前台）
bash run_all.sh

# 后台运行
nohup bash run_all.sh > full.log 2>&1 &

# 查看进度
./check_progress.sh
tail -f full.log
```

## 使用训练数据

```python
import pickle

# 加载数据
with open('output/train_data.pkl', 'rb') as f:
    data = pickle.load(f)

train_data = data['data']      # 194,672个样本
features = data['feature_names'] # 49个特征

# 查看样本
sample = train_data[0]
print(sample['values'].shape)   # (48, 49)
print(sample['masks'].shape)    # (48, 49)
print(sample['deltas'].shape)   # (48, 49)
print(sample['readmit_30d'])    # 0 or 1
```
