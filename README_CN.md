# 信贷模型Python代码

### 环境配置
- 请将代码根目录credit_model放到\
~/workspace/services

- 请将数据集放到\
~/workspace/data


- 请创建python 3.10的虚拟环境


- 请在credit_model的根目录下依次运行下面指令
1. conda install -c conda-forge toad\
2. conda install -c anaconda statsmodels
3. conda install -c conda-forge matplotlib\
4. conda install -c conda-forge lightgbm\
5. pip install -r requirements.txt

如果出现PIL的错误 请按下述步骤操作
pip uninstall pillow
pip install pillow==9.4.0

在Windows上安装scoop并安装graphviz\
check doc: https://github.com/ScoopInstaller/Scoop \
scoop install graphviz

### Demo
demo文件夹下的notebooks是构建信贷模型的基础模块代码样例

### Projects
prjects文件夹下的代码脚本是ABC卡的代码样例
请根据文件夹下的step1到step5依次运行代码