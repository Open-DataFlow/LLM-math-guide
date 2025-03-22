

## 安装Llama-Factory
1. 官方仓库：https://github.com/hiyouga/LLaMA-Factory
2. 安装命令，一般找一个工作路径clone下来安装即可：
```
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
3. 测试是否安装正确：
```
llamafactory-cli version
```
如果安装正确的话，会报告版本信息以及官方github仓库链接。

## 准备